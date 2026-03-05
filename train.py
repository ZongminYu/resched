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
from typing import Dict, List, Tuple

from config import Config, EnvConfig, ModelConfig, TrainConfig
from env.fjsp_env import FJSPEnv
from env.instance_generator import InstanceGenerator
from model.network import ReSchedPolicy
from utils import set_seed, AverageMeter


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
    """Validate with greedy decoding."""
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

    # Training
    os.makedirs(train_config.save_dir, exist_ok=True)
    best_val_makespan = float('inf')

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
            val_makespan = validate(
                policy, env, generator, train_config.val_size, device
            )
            print(f"  [Validation] Makespan: {val_makespan:.2f}")

            if val_makespan < best_val_makespan:
                best_val_makespan = val_makespan
                save_path = os.path.join(
                    train_config.save_dir,
                    f'resched_{env_config.dataset_type}_{env_config.num_jobs}x{env_config.num_machines}_best.pt'
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

    print(f"\nTraining complete. Best validation makespan: {best_val_makespan:.2f}")


if __name__ == '__main__':
    main()
