"""
ReSched Policy Network.

Dual-branch Transformer architecture:
  - Operation Branch: Self-attention with RoPE for O2O dependencies.
  - Machine Branch: Cross-attention with Edge-in-Attention + Self-based Cross-Attention.
  - Decision-Making: MLP scoring over feasible operation-machine pairs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
from .rope import RotaryPositionalEmbedding


class OperationBranch(nn.Module):
    """
    Operation branch: Self-attention with RoPE.
    Models O2O (intra-job) dependencies among operations using self-attention.

    RoPE is applied within each job to encode relative positions.
    Attention is masked so that each operation only attends to itself and its
    successors within the same job (backward-looking hop connections).
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim)

        # Layer norm and FFN
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        o2o_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) operation embeddings.
            o2o_mask: (B, N, N) bool mask for O2O attention.
            positions: (B, N) intra-job position indices.

        Returns:
            x: (B, N, D) updated operation embeddings.
        """
        B, N, D = x.shape
        H = self.num_heads
        d_k = self.head_dim

        # Self-attention with RoPE
        residual = x
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        k = self.k_proj(x_norm).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        v = self.v_proj(x_norm).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)

        # Apply RoPE
        q, k = self.rope(q, k, positions)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, N, N)

        # Apply O2O mask (positions that should not attend get -inf)
        attn_mask = o2o_mask.unsqueeze(1)  # (B, 1, N, N)
        scores = scores.masked_fill(~attn_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        # Handle NaN from all-masked rows
        attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0.0)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # (B, H, N, d_k)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        attn_out = self.out_proj(attn_out)

        x = residual + self.dropout(attn_out)

        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class MachineBranch(nn.Module):
    """
    Machine branch: Cross-attention with Edge-in-Attention + Self-based Cross-attention.

    Each machine attends to all operations connected to it (via O2M),
    incorporating edge features (duration) directly into Q, K, V.
    Self-based cross-attention: machine also attends to its own embedding.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # Query from machine, Key/Value from operations (cross-attention)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Independent K/V projections for machine self-attention (Eq. 6)
        # Separate from operation K/V to maintain distinct parameter spaces
        self.k_self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_self_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge embedding: shared projection for edge features (duration)
        # Shared across Q, K, V and all heads for efficiency
        self.edge_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norm and FFN
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm_op = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        machine_emb: torch.Tensor,
        op_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        o2m_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            machine_emb: (B, M, D) machine embeddings.
            op_emb: (B, N, D) operation embeddings.
            edge_emb: (B, N, M, D) edge embeddings (from duration).
            o2m_mask: (B, N, M) bool mask for O2M connections.

        Returns:
            machine_emb: (B, M, D) updated machine embeddings.
        """
        B, M, D = machine_emb.shape
        N = op_emb.shape[1]
        H = self.num_heads
        d_k = self.head_dim

        residual = machine_emb
        machine_norm = self.norm1(machine_emb)
        op_norm = self.norm_op(op_emb)

        # Machine queries: (B, M, D) -> (B, H, M, d_k)
        q_m = self.q_proj(machine_norm).view(B, M, H, d_k).transpose(1, 2)  # (B, H, M, d_k)

        # Operation keys and values: (B, N, D) -> (B, H, N, d_k)
        k_o = self.k_proj(op_norm).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
        v_o = self.v_proj(op_norm).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)

        # Edge projections: shared across heads and Q/K/V
        # edge_emb: (B, N, M, D) -> (B, N, M, D)
        edge_proj = self.edge_proj(edge_emb)  # (B, N, M, D)
        # Reshape for heads: (B, N, M, H, d_k) -> (B, H, N, M, d_k)
        edge_proj = edge_proj.view(B, N, M, H, d_k).permute(0, 3, 1, 2, 4)

        # Edge-in-Attention (Eq. 5):
        # For each machine m and operation o:
        # q = q_m[m] + edge_q[o,m], k = k_o[o] + edge_k[o,m], v = v_o[o] + edge_v[o,m]
        # We share the same edge projection for q, k, v

        # Compute attention: for each machine m, attend to all connected operations
        # q_m: (B, H, M, d_k) -> (B, H, M, 1, d_k)
        q_m_exp = q_m.unsqueeze(3)  # (B, H, M, 1, d_k)

        # edge for query: (B, H, N, M, d_k) -> permute to (B, H, M, N, d_k)
        edge_for_attn = edge_proj.permute(0, 1, 3, 2, 4)  # (B, H, M, N, d_k)

        # q_total = q_m + edge (for each machine-operation pair)
        # (B, H, M, 1, d_k) + (B, H, M, N, d_k) -> (B, H, M, N, d_k)
        q_total = q_m_exp + edge_for_attn

        # k_total: k_o + edge
        # k_o: (B, H, N, d_k) -> (B, H, 1, N, d_k)
        k_o_exp = k_o.unsqueeze(2)  # (B, H, 1, N, d_k)
        k_total = k_o_exp + edge_for_attn  # (B, H, M, N, d_k)

        # v_total: v_o + edge
        v_o_exp = v_o.unsqueeze(2)  # (B, H, 1, N, d_k)
        v_total = v_o_exp + edge_for_attn  # (B, H, M, N, d_k)

        # Attention scores: (B, H, M, N)
        scores = (q_total * k_total).sum(dim=-1) / math.sqrt(d_k)  # (B, H, M, N)

        # Self-based Cross-attention (Eq. 6):
        # Machine attends to its own representation with independent K projection
        k_self = self.k_self_proj(machine_norm).view(B, M, H, d_k).transpose(1, 2)  # (B, H, M, d_k)
        self_scores = (q_m * k_self).sum(dim=-1)  # (B, H, M)
        self_scores = self_scores / math.sqrt(d_k)

        # O2M mask: (B, N, M) -> (B, 1, M, N)
        cross_mask = o2m_mask.permute(0, 2, 1).unsqueeze(1)  # (B, 1, M, N)

        # Concatenate self-score with cross-scores
        # scores: (B, H, M, N), self_scores: (B, H, M)
        # Combined: (B, H, M, N+1)
        combined_scores = torch.cat([self_scores.unsqueeze(-1), scores], dim=-1)

        # Mask: self always visible, others via O2M
        self_mask_flag = torch.ones(B, 1, M, 1, dtype=torch.bool, device=machine_emb.device)
        combined_mask = torch.cat([self_mask_flag, cross_mask], dim=-1)  # (B, 1, M, N+1)

        combined_scores = combined_scores.masked_fill(~combined_mask, float('-inf'))
        attn_weights = F.softmax(combined_scores, dim=-1)  # (B, H, M, N+1)
        attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0.0)
        attn_weights = self.dropout(attn_weights)

        # Self-value: independent V projection for machine self-attention
        v_self = self.v_self_proj(machine_norm).view(B, M, H, d_k).transpose(1, 2)  # (B, H, M, d_k)

        # Combined values: (B, H, M, N+1, d_k)
        combined_values = torch.cat([v_self.unsqueeze(3), v_total], dim=3)  # (B, H, M, N+1, d_k)

        # Weighted sum
        attn_out = torch.einsum('bhmi,bhmid->bhmd', attn_weights, combined_values)  # (B, H, M, d_k)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, D)  # (B, M, D)
        attn_out = self.out_proj(attn_out)

        machine_emb = residual + self.dropout(attn_out)

        # FFN
        residual = machine_emb
        machine_emb = residual + self.ffn(self.norm2(machine_emb))

        return machine_emb


class FeatureExtraction(nn.Module):
    """
    Feature extraction network with dual branches:
      - Operation branch (self-attention + RoPE)
      - Machine branch (cross-attention + edge + self-based)
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.num_layers = num_layers

        # Input embeddings
        # Operation: available_time + min_duration -> hidden_dim
        self.op_embed = nn.Linear(2, hidden_dim)
        # Machine: available_time -> hidden_dim
        self.machine_embed = nn.Linear(1, hidden_dim)
        # Edge (duration): scalar -> hidden_dim
        self.edge_embed = nn.Linear(1, hidden_dim)

        # Stacked layers
        self.op_layers = nn.ModuleList([
            OperationBranch(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.machine_layers = nn.ModuleList([
            MachineBranch(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Dict from FJSPEnv._get_state()

        Returns:
            op_emb: (B, N, D) operation embeddings
            machine_emb: (B, M, D) machine embeddings
            edge_emb: (B, N, M, D) edge embeddings
        """
        # Extract features
        op_avail = state['op_available_time']  # (B, N)
        min_dur = state['min_duration']         # (B, N)
        m_avail = state['machine_available_time']  # (B, M)
        dur_matrix = state['duration_matrix']   # (B, N, M)
        o2o_mask = state['o2o_mask']             # (B, N, N)
        o2m_mask = state['o2m_mask']             # (B, N, M)
        positions = state['op_positions']         # (B, N)
        op_mask = state['op_mask']               # (B, N)

        # Initial embeddings
        # Operation: concatenate available_time and min_duration
        op_features = torch.stack([op_avail, min_dur], dim=-1)  # (B, N, 2)
        op_emb = self.op_embed(op_features)  # (B, N, D)

        # Machine: available_time
        m_features = m_avail.unsqueeze(-1)  # (B, M, 1)
        machine_emb = self.machine_embed(m_features)  # (B, M, D)

        # Edge: duration
        edge_features = dur_matrix.unsqueeze(-1)  # (B, N, M, 1)
        edge_emb = self.edge_embed(edge_features)  # (B, N, M, D)

        # Mask out padded operations in embeddings
        op_emb = op_emb * op_mask.unsqueeze(-1).float()

        # Apply layers
        for op_layer, m_layer in zip(self.op_layers, self.machine_layers):
            op_emb = op_layer(op_emb, o2o_mask, positions)
            op_emb = op_emb * op_mask.unsqueeze(-1).float()

            machine_emb = m_layer(machine_emb, op_emb, edge_emb, o2m_mask)

        return op_emb, machine_emb, edge_emb


class DecisionMaking(nn.Module):
    """
    Decision-making MLP.
    Takes operation and machine embeddings + edge embedding for each feasible pair,
    and produces a scalar score. Softmax over scores yields action probabilities.
    """

    def __init__(self, hidden_dim: int, mlp_hidden_dim: int = 64, mlp_num_layers: int = 3):
        super().__init__()
        # Input: op_emb + machine_emb + edge_emb = 3 * hidden_dim
        layers = []
        in_dim = 3 * hidden_dim
        for i in range(mlp_num_layers - 1):
            layers.append(nn.Linear(in_dim, mlp_hidden_dim))
            layers.append(nn.ReLU())
            in_dim = mlp_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        op_emb: torch.Tensor,
        machine_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            op_emb: (B, N, D)
            machine_emb: (B, M, D)
            edge_emb: (B, N, M, D)
            action_mask: (B, N, M) bool

        Returns:
            log_probs: (B, N*M) log probabilities over flattened action space.
        """
        B, N, D = op_emb.shape
        M = machine_emb.shape[1]

        # Expand for all pairs: (B, N, M, D)
        op_exp = op_emb.unsqueeze(2).expand(B, N, M, D)
        m_exp = machine_emb.unsqueeze(1).expand(B, N, M, D)

        # Concatenate: (B, N, M, 3*D)
        pair_features = torch.cat([op_exp, m_exp, edge_emb], dim=-1)

        # MLP scores: (B, N, M, 1) -> (B, N, M)
        scores = self.mlp(pair_features).squeeze(-1)

        # Mask infeasible actions
        scores = scores.masked_fill(~action_mask, float('-inf'))

        # Flatten: (B, N*M)
        scores_flat = scores.view(B, -1)

        # Log softmax
        log_probs = F.log_softmax(scores_flat, dim=-1)

        return log_probs


class ReSchedPolicy(nn.Module):
    """
    Full ReSched policy network.
    Combines feature extraction and decision-making modules.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        ffn_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_hidden_dim: int = 64,
        mlp_num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_extraction = FeatureExtraction(
            hidden_dim, num_heads, ffn_dim, num_layers, dropout
        )
        self.decision_making = DecisionMaking(
            hidden_dim, mlp_hidden_dim, mlp_num_layers
        )
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: Dict from FJSPEnv._get_state()

        Returns:
            log_probs: (B, N*M) log probabilities over action space.
            action_mask_flat: (B, N*M) bool mask of feasible actions.
        """
        op_emb, machine_emb, edge_emb = self.feature_extraction(state)

        action_mask = state['action_mask']
        log_probs = self.decision_making(op_emb, machine_emb, edge_emb, action_mask)

        B = action_mask.shape[0]
        action_mask_flat = action_mask.view(B, -1)

        return log_probs, action_mask_flat

    def select_action(self, state: Dict, greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action from the policy.

        Args:
            state: Dict from FJSPEnv._get_state()
            greedy: if True, select argmax; otherwise sample.

        Returns:
            actions: (B, 2) tensor [op_idx, machine_idx]
            log_probs: (B,) log probability of selected actions
        """
        log_probs_all, action_mask_flat = self.forward(state)

        B = log_probs_all.shape[0]
        N = state['action_mask'].shape[1]
        M = state['action_mask'].shape[2]

        if greedy:
            # Select action with highest probability
            flat_indices = log_probs_all.argmax(dim=-1)  # (B,)
        else:
            # Sample from categorical distribution
            probs = log_probs_all.exp()
            # Replace any inf/nan with 0
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            probs = probs.clamp(min=0.0)
            # For done instances all probs are 0; fall back to uniform to avoid multinomial crash
            prob_sum = probs.sum(dim=-1, keepdim=True)  # (B, 1)
            zero_instance = (prob_sum == 0).expand_as(probs)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = torch.where(zero_instance, uniform, probs / prob_sum.clamp(min=1e-8))
            flat_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

        # Convert flat index to (op, machine) pair
        op_indices = flat_indices // M
        m_indices = flat_indices % M

        actions = torch.stack([op_indices, m_indices], dim=-1)  # (B, 2)

        # Gather log probs for selected actions
        selected_log_probs = log_probs_all.gather(1, flat_indices.unsqueeze(1)).squeeze(1)  # (B,)

        return actions, selected_log_probs
