"""
Rotary Positional Embedding (RoPE) for ReSched.

Applied within the operation branch to encode intra-job relative positions.
See: Su et al., 2024 - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""
import torch
import torch.nn as nn
import math


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    Applies rotation to query and key vectors based on their positions,
    enabling the attention to be a function of relative position.
    """

    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        # Compute inverse frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def _compute_sin_cos(self, positions: torch.Tensor) -> tuple:
        """
        Compute sin and cos for given positions.

        Args:
            positions: (B, N) integer position indices.

        Returns:
            sin, cos: (B, N, dim) tensors.
        """
        # positions: (B, N)
        # inv_freq: (dim/2,)
        # sinusoid: (B, N, dim/2)
        sinusoid = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)
        sin = sinusoid.sin()  # (B, N, dim/2)
        cos = sinusoid.cos()  # (B, N, dim/2)
        return sin, cos

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """
        Apply RoPE to query and key tensors.

        Args:
            q: (B, H, N, D) query tensor
            k: (B, H, N, D) key tensor
            positions: (B, N) position indices for each token (intra-job position)

        Returns:
            q_rot, k_rot: rotated query and key tensors.
        """
        B, H, N, D = q.shape
        sin, cos = self._compute_sin_cos(positions)  # (B, N, D/2)

        # Reshape for broadcasting with heads
        sin = sin.unsqueeze(1)  # (B, 1, N, D/2)
        cos = cos.unsqueeze(1)  # (B, 1, N, D/2)

        q_rot = self._apply_rotation(q, sin, cos)
        k_rot = self._apply_rotation(k, sin, cos)

        return q_rot, k_rot

    def _apply_rotation(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to tensor x using sin and cos.

        Args:
            x: (B, H, N, D)
            sin: (B, 1, N, D/2)
            cos: (B, 1, N, D/2)

        Returns:
            rotated x: (B, H, N, D)
        """
        D = x.shape[-1]
        x1 = x[..., :D // 2]  # (B, H, N, D/2)
        x2 = x[..., D // 2:]  # (B, H, N, D/2)

        # Apply rotation
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos

        return torch.cat([rot_x1, rot_x2], dim=-1)
