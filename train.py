"""
Training script for ReSched using REINFORCE algorithm.

Algorithm 1 from the paper:
  - Generate batch of instances
  - Roll out policy to get trajectory
  - Compute discounted returns
  - Normalize advantages
  - Update policy via REINFORCE
"""
import os
import time
import argparse
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import Config, EnvConfig, ModelConfig, TrainConfig
from env.fjsp_env import FJSPEnv
from env.instance_generator import InstanceGenerator
from model.network import ReSchedPolicy
from utils import set_seed, AverageMeter


# ---------------------------------------------------------------------------
# JSSP helpers (same data format as TBGAT / n-step_reinforce.py)
# ---------------------------------------------------------------------------

def jssp_batch_to_resched_instance(
    times_batch: np.ndarray,
    machines_batch: np.ndarray,
    device: torch.device,
) -> Dict:
    """
    Convert a batch of TBGAT-style JSSP instances to resched instance dict.

    Args:
        times_batch:   (B, n_j, n_m) int/float  – processing times
        machines_batch:(B, n_j, n_m) int        – machine assignments (1-indexed)
        device: torch device

    Returns:
        Instance dict compatible with FJSPEnv.reset()
    """
    B, n_j, n_m = times_batch.shape
    total_ops = n_j * n_m

    # Flat operation ordering: op_idx = job * n_m + position_within_job
    jobs_arr      = np.repeat(np.arange(n_j), n_m)          # (total_ops,)
    positions_arr = np.tile(np.arange(n_m), n_j)            # (total_ops,)

    times_flat    = times_batch.reshape(B, total_ops).astype(np.float32)   # (B, T)
    machines_flat = machines_batch.reshape(B, total_ops).astype(np.int64) - 1  # (B, T), 0-indexed

    # Duration matrix and O2M mask via vectorised fancy indexing
    duration_matrix = np.zeros((B, total_ops, n_m), dtype=np.float32)
    o2m_mask_np     = np.zeros((B, total_ops, n_m), dtype=bool)
    b_idx = np.arange(B)[:, None]           # (B, 1)
    t_idx = np.arange(total_ops)[None, :]   # (1, T)
    duration_matrix[b_idx, t_idx, machines_flat] = times_flat
    o2m_mask_np[b_idx, t_idx, machines_flat]     = True

    # O2O mask: within each job, earlier ops precede later ops (shared across batch)
    o2o_single = np.zeros((total_ops, total_ops), dtype=bool)
    for j in range(n_j):
        start = j * n_m
        for i in range(n_m):
            for k in range(i + 1, n_m):
                o2o_single[start + i, start + k] = True
    o2o_batch = np.broadcast_to(o2o_single[None], (B, total_ops, total_ops)).copy()

    return {
        'duration_matrix': torch.from_numpy(duration_matrix).to(device),
        'job_indices':      torch.from_numpy(
            np.broadcast_to(jobs_arr[None], (B, total_ops)).copy()
        ).long().to(device),
        'op_positions':     torch.from_numpy(
            np.broadcast_to(positions_arr[None], (B, total_ops)).copy()
        ).long().to(device),
        'op_mask':          torch.ones(B, total_ops, dtype=torch.bool, device=device),
        'o2m_mask':         torch.from_numpy(o2m_mask_np).to(device),
        'o2o_mask':         torch.from_numpy(o2o_batch).to(device),
        # For JSSP each op has exactly one machine, so min_duration == duration
        'min_duration':     torch.from_numpy(times_flat).to(device),
        'num_ops_per_job':  torch.full((B, n_j), n_m, dtype=torch.long, device=device),
        'total_ops':        torch.full((B,), total_ops, dtype=torch.long, device=device),
        'num_machines':     n_m,
    }


def load_jssp_validation_data(
    num_jobs: int,
    num_machines: int,
    device: torch.device,
) -> Tuple[Dict, np.ndarray]:
    """
    Load the pre-computed JSSP validation dataset and optimal Cmax values
    (same files used by TBGAT / n-step_reinforce.py).

    Searches in order:
      1. <repo_root>/TBGAT/validation_data/
      2. ./validation_data/

    Returns:
        (instance_dict, validation_Cmax)
        instance_dict     – resched-format instance dict for env.reset()
        validation_Cmax   – ndarray (N,) of CP-SAT optimal makespan per instance
    """
    fname = f'JSSP_validation_data_and_Cmax_{num_jobs}x{num_machines}_[1,99].npy'
    repo_root   = Path(__file__).resolve().parent.parent
    search_paths = [
        repo_root / 'TBGAT' / 'validation_data' / fname,
        Path(__file__).resolve().parent / 'validation_data' / fname,
    ]

    data = None
    for p in search_paths:
        if p.is_file():
            data = np.load(str(p))   # shape: (#inst, 3, n_j, n_m)
            print(f'Loaded JSSP validation data from {p}')
            break

    if data is None:
        raise FileNotFoundError(
            f'JSSP validation data not found. Searched:\n'
            + '\n'.join(str(p) for p in search_paths)
        )

    times    = data[:, 0, :, :]              # (N, n_j, n_m)
    machines = data[:, 1, :, :]              # (N, n_j, n_m), 1-indexed
    cmax     = data[:, 2, 0, 0].astype(float)  # (N,)

    print(f'  {times.shape[0]} validation instances, mean Cmax = {cmax.mean():.2f}')
    instance_dict = jssp_batch_to_resched_instance(times, machines, device)
    return instance_dict, cmax


def validate_jssp(
    policy: 'ReSchedPolicy',
    env: FJSPEnv,
    val_instance: Dict,
    val_cmax: np.ndarray,
) -> Tuple[float, float]:
    """
    Validate on the fixed JSSP validation set and report optimality gap.

    Returns:
        gap_incumbent:  mean gap of best makespan found during rollout
        gap_last_step:  mean gap of makespan at final step
    """
    policy.eval()
    with torch.no_grad():
        _, _, makespan = rollout(env, policy, val_instance, greedy=True)
    makespan_np = makespan.cpu().numpy()       # (N,)
    gap = ((makespan_np - val_cmax) / val_cmax).mean()
    return gap


# ---------------------------------------------------------------------------


def rollout(
    env: FJSPEnv,
    policy: ReSchedPolicy,
    instance: Dict,
    greedy: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Roll out the policy on a batch of instances.

    Returns:
        log_probs_list: list of (B,) log prob tensors at each step.
        rewards_list: list of (B,) reward tensors at each step.
        makespan: (B,) final makespan.
    """
    state = env.reset(instance)
    log_probs_list = []
    rewards_list = []

    while not env.done.all():
        actions, log_probs = policy.select_action(state, greedy=greedy)
        next_state, rewards, done = env.step(actions)
        log_probs_list.append(log_probs)
        rewards_list.append(rewards)
        state = next_state

    makespan = env.get_makespan()
    return log_probs_list, rewards_list, makespan


def compute_returns(rewards_list: List[torch.Tensor], gamma: float) -> torch.Tensor:
    """
    Compute discounted cumulative returns.

    Args:
        rewards_list: list of T tensors, each (B,).
        gamma: discount factor.

    Returns:
        returns: (T, B) tensor of discounted returns.
    """
    T = len(rewards_list)
    B = rewards_list[0].shape[0]
    device = rewards_list[0].device

    returns = torch.zeros(T, B, device=device)
    running_return = torch.zeros(B, device=device)

    for t in reversed(range(T)):
        running_return = rewards_list[t] + gamma * running_return
        returns[t] = running_return

    return returns


def train_epoch(
    policy: ReSchedPolicy,
    optimizer: optim.Optimizer,
    env: FJSPEnv,
    generator: InstanceGenerator,
    config: TrainConfig,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        avg_loss: average policy loss.
        avg_makespan: average makespan.
    """
    policy.train()
    loss_meter = AverageMeter()
    makespan_meter = AverageMeter()

    num_batches = config.instances_per_epoch // config.batch_size

    for batch_idx in range(num_batches):
        # Generate batch
        instance = generator.generate(config.batch_size, device=device)

        # Rollout
        log_probs_list, rewards_list, makespan = rollout(env, policy, instance, greedy=False)

        # Compute returns
        returns = compute_returns(rewards_list, config.gamma)  # (T, B)

        # Normalize advantages
        advantages = returns - returns.mean(dim=1, keepdim=True)
        std = returns.std(dim=1, keepdim=True)
        std = std.clamp(min=1e-8)
        advantages = advantages / std

        # Compute policy loss: -sum_t A_t * log_pi(a_t|s_t)
        log_probs = torch.stack(log_probs_list, dim=0)  # (T, B)
        policy_loss = -(advantages.detach() * log_probs).mean()

        # Update
        optimizer.zero_grad()
        policy_loss.backward()
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
        optimizer.step()

        loss_meter.update(policy_loss.item(), config.batch_size)
        makespan_meter.update(makespan.mean().item(), config.batch_size)

        # Explicitly release the trajectory buffers so the computation graphs
        # (one full forward pass per step) are freed before the next batch.
        del log_probs_list, rewards_list, log_probs, returns, advantages, \
            policy_loss, makespan, instance

    return loss_meter.avg, makespan_meter.avg


def validate(
    policy: ReSchedPolicy,
    env: FJSPEnv,
    generator: InstanceGenerator,
    val_size: int,
    device: torch.device,
) -> float:
    """Validate with greedy decoding (non-JSSP datasets)."""
    policy.eval()
    with torch.no_grad():
        instance = generator.generate(val_size, device=device)
        _, _, makespan = rollout(env, policy, instance, greedy=True)
    return makespan.mean().item()


def main():
    parser = argparse.ArgumentParser(description='Train ReSched')
    parser.add_argument('--num_jobs', type=int, default=10)
    parser.add_argument('--num_machines', type=int, default=5)
    parser.add_argument('--dataset_type', type=str, default='SD1', choices=['SD1', 'SD2', 'JSSP', 'FFSP'])
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--instances_per_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--val_freq', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=1)
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Config
    env_config = EnvConfig(
        num_jobs=args.num_jobs,
        num_machines=args.num_machines,
        dataset_type=args.dataset_type,
    )
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    train_config = TrainConfig(
        lr=args.lr,
        gamma=args.gamma,
        num_epochs=args.num_epochs,
        instances_per_epoch=args.instances_per_epoch,
        batch_size=args.batch_size,
        device=args.device,
        save_dir=args.save_dir,
        val_freq=args.val_freq,
        log_freq=args.log_freq,
    )

    # Instance generator
    generator = InstanceGenerator(
        num_jobs=env_config.num_jobs,
        num_machines=env_config.num_machines,
        ops_low=env_config.ops_low,
        ops_high=env_config.ops_high,
        duration_low=env_config.duration_low,
        duration_high=env_config.duration_high,
        dataset_type=env_config.dataset_type,
    )

    # Environment
    env = FJSPEnv(device=device)

    # Policy network
    policy = ReSchedPolicy(
        hidden_dim=model_config.hidden_dim,
        ffn_dim=model_config.ffn_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        mlp_hidden_dim=model_config.mlp_hidden_dim,
        mlp_num_layers=model_config.mlp_num_layers,
        dropout=model_config.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy network parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=train_config.lr)

    # JSSP: load fixed validation dataset + optimal Cmax (same as n-step_reinforce.py)
    is_jssp = env_config.dataset_type == 'JSSP'
    jssp_val_instance: Optional[Dict] = None
    jssp_val_cmax: Optional[np.ndarray] = None
    if is_jssp:
        jssp_val_instance, jssp_val_cmax = load_jssp_validation_data(
            env_config.num_jobs, env_config.num_machines, device
        )

    # Training
    os.makedirs(train_config.save_dir, exist_ok=True)
    best_val_makespan = float('inf')
    best_val_gap      = float('inf')   # used for JSSP

    print(f"\n{'='*60}")
    print(f"Training ReSched on {env_config.dataset_type} "
          f"({env_config.num_jobs}x{env_config.num_machines})")
    print(f"Epochs: {train_config.num_epochs}, "
          f"Instances/Epoch: {train_config.instances_per_epoch}, "
          f"Batch: {train_config.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, train_config.num_epochs + 1):
        t0 = time.time()
        avg_loss, avg_makespan = train_epoch(
            policy, optimizer, env, generator, train_config, device
        )
        epoch_time = time.time() - t0

        if epoch % train_config.log_freq == 0:
            print(f"Epoch {epoch:4d}/{train_config.num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Makespan: {avg_makespan:.2f} | "
                  f"Time: {epoch_time:.1f}s")

        # Validation
        if epoch % train_config.val_freq == 0:
            if is_jssp:
                # ---- JSSP: fixed dataset + optimality gap (same as n-step_reinforce.py) ----
                val_gap = validate_jssp(policy, env, jssp_val_instance, jssp_val_cmax)
                print(f"  [Validation] Gap: {val_gap:.6f}")

                if val_gap < best_val_gap:
                    best_val_gap = val_gap
                    save_path = os.path.join(
                        train_config.save_dir,
                        f'resched_{env_config.dataset_type}_'
                        f'{env_config.num_jobs}x{env_config.num_machines}_best.pt'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_gap': val_gap,
                        'config': {
                            'env': env_config.__dict__,
                            'model': model_config.__dict__,
                            'train': train_config.__dict__,
                        },
                    }, save_path)
                    print(f"  [Saved] New best model (gap: {val_gap:.6f})")
            else:
                # ---- non-JSSP: on-the-fly generation, raw makespan ----
                val_makespan = validate(
                    policy, env, generator, train_config.val_size, device
                )
                print(f"  [Validation] Makespan: {val_makespan:.2f}")

                if val_makespan < best_val_makespan:
                    best_val_makespan = val_makespan
                    save_path = os.path.join(
                        train_config.save_dir,
                        f'resched_{env_config.dataset_type}_'
                        f'{env_config.num_jobs}x{env_config.num_machines}_best.pt'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_makespan': val_makespan,
                        'config': {
                            'env': env_config.__dict__,
                            'model': model_config.__dict__,
                            'train': train_config.__dict__,
                        },
                    }, save_path)
                    print(f"  [Saved] New best model (makespan: {val_makespan:.2f})")

    if is_jssp:
        print(f"\nTraining complete. Best validation gap: {best_val_gap:.6f}")
    else:
        print(f"\nTraining complete. Best validation makespan: {best_val_makespan:.2f}")


if __name__ == '__main__':
    main()
