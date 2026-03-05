"""
Evaluation script for ReSched.

Supports:
  - Greedy decoding
  - Sampling strategy (100 trajectories, report best)
  - Evaluation on synthetic data and benchmarks
"""
import os
import argparse
import torch
import numpy as np
import time
from typing import Dict, List, Tuple

from config import EnvConfig, ModelConfig
from env.fjsp_env import FJSPEnv
from env.instance_generator import InstanceGenerator
from model.network import ReSchedPolicy
from utils import set_seed


def evaluate_greedy(
    policy: ReSchedPolicy,
    env: FJSPEnv,
    instances: Dict,
    device: torch.device,
) -> torch.Tensor:
    """Evaluate policy with greedy decoding."""
    policy.eval()
    with torch.no_grad():
        state = env.reset(instances)
        while not env.done.all():
            actions, _ = policy.select_action(state, greedy=True)
            state, _, _ = env.step(actions)
    return env.get_makespan()


def evaluate_sampling(
    policy: ReSchedPolicy,
    env: FJSPEnv,
    instances: Dict,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Evaluate policy with sampling strategy.
    For each instance, run num_samples stochastic trajectories and report the best.
    """
    policy.eval()
    batch_size = instances['duration_matrix'].shape[0]
    best_makespan = torch.full((batch_size,), float('inf'), device=device)

    with torch.no_grad():
        for s in range(num_samples):
            state = env.reset(instances)
            while not env.done.all():
                actions, _ = policy.select_action(state, greedy=False)
                state, _, _ = env.step(actions)
            makespan = env.get_makespan()
            best_makespan = torch.min(best_makespan, makespan)

    return best_makespan


def load_benchmark_instance(filepath: str, device: torch.device) -> Dict:
    """
    Load a benchmark FJSP instance from standard file format.

    Standard FJSP format:
      Line 1: num_jobs num_machines (avg_machines_per_op)
      Each subsequent line defines a job:
        num_operations  [num_machines_for_op1  machine1 duration1 machine2 duration2 ...] [...]

    Returns a single-instance dict compatible with FJSPEnv.reset().
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    first_line = lines[0].split()
    num_jobs = int(first_line[0])
    num_machines = int(first_line[1])

    all_ops = []
    job_indices = []
    op_positions = []
    duration_matrices = []
    o2m_masks = []

    for job_id in range(num_jobs):
        line = lines[job_id + 1].split()
        idx = 0
        num_ops = int(line[idx])
        idx += 1

        for pos in range(num_ops):
            num_eligible = int(line[idx])
            idx += 1
            dur_row = np.zeros(num_machines, dtype=np.float32)
            mask_row = np.zeros(num_machines, dtype=bool)

            for _ in range(num_eligible):
                machine_id = int(line[idx]) - 1  # 0-indexed
                duration = float(line[idx + 1])
                dur_row[machine_id] = duration
                mask_row[machine_id] = True
                idx += 2

            duration_matrices.append(dur_row)
            o2m_masks.append(mask_row)
            job_indices.append(job_id)
            op_positions.append(pos)

    total_ops = len(duration_matrices)
    duration_matrix = np.stack(duration_matrices)  # (total_ops, num_machines)
    o2m_mask = np.stack(o2m_masks)                  # (total_ops, num_machines)
    job_indices_arr = np.array(job_indices, dtype=np.int64)
    op_positions_arr = np.array(op_positions, dtype=np.int64)

    # Min duration
    min_duration = np.zeros(total_ops, dtype=np.float32)
    for i in range(total_ops):
        feasible = duration_matrix[i][o2m_mask[i]]
        if len(feasible) > 0:
            min_duration[i] = feasible.min()

    # O2O mask (backward hop connections)
    o2o_mask = np.zeros((total_ops, total_ops), dtype=bool)
    op_idx = 0
    ops_per_job = []
    current_job = 0
    count = 0
    for i, j_id in enumerate(job_indices_arr):
        if j_id != current_job:
            ops_per_job.append(count)
            current_job = j_id
            count = 0
        count += 1
    ops_per_job.append(count)

    op_idx = 0
    for job_id in range(num_jobs):
        n_ops = ops_per_job[job_id]
        for i in range(n_ops):
            for j in range(i + 1, n_ops):
                o2o_mask[op_idx + i, op_idx + j] = True
        op_idx += n_ops

    # Convert to tensors and add batch dim
    instance = {
        'duration_matrix': torch.from_numpy(duration_matrix).unsqueeze(0).to(device),
        'job_indices': torch.from_numpy(job_indices_arr).unsqueeze(0).to(device),
        'op_positions': torch.from_numpy(op_positions_arr).unsqueeze(0).to(device),
        'op_mask': torch.ones(1, total_ops, dtype=torch.bool, device=device),
        'o2m_mask': torch.from_numpy(o2m_mask).unsqueeze(0).to(device),
        'o2o_mask': torch.from_numpy(o2o_mask).unsqueeze(0).to(device),
        'min_duration': torch.from_numpy(min_duration).unsqueeze(0).to(device),
        'num_ops_per_job': torch.tensor([ops_per_job], dtype=torch.long, device=device),
        'total_ops': torch.tensor([total_ops], dtype=torch.long, device=device),
        'num_machines': num_machines,
    }

    return instance


def main():
    parser = argparse.ArgumentParser(description='Evaluate ReSched')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--strategy', type=str, default='greedy', choices=['greedy', 'sampling'])
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for sampling strategy')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    # Evaluation mode
    parser.add_argument('--mode', type=str, default='synthetic', choices=['synthetic', 'benchmark'])

    # Synthetic evaluation
    parser.add_argument('--num_jobs', type=int, default=10)
    parser.add_argument('--num_machines', type=int, default=5)
    parser.add_argument('--dataset_type', type=str, default='SD1')
    parser.add_argument('--num_instances', type=int, default=100)

    # Benchmark evaluation
    parser.add_argument('--benchmark_dir', type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get('config', {}).get('model', {})

    policy = ReSchedPolicy(
        hidden_dim=model_config.get('hidden_dim', 128),
        ffn_dim=model_config.get('ffn_dim', 512),
        num_heads=model_config.get('num_heads', 8),
        num_layers=model_config.get('num_layers', 2),
        mlp_hidden_dim=model_config.get('mlp_hidden_dim', 64),
        mlp_num_layers=model_config.get('mlp_num_layers', 3),
    ).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()

    env = FJSPEnv(device=device)

    if args.mode == 'synthetic':
        # Generate synthetic test instances
        env_config = EnvConfig(
            num_jobs=args.num_jobs,
            num_machines=args.num_machines,
            dataset_type=args.dataset_type,
        )
        generator = InstanceGenerator(
            num_jobs=env_config.num_jobs,
            num_machines=env_config.num_machines,
            ops_low=env_config.ops_low,
            ops_high=env_config.ops_high,
            duration_low=env_config.duration_low,
            duration_high=env_config.duration_high,
            dataset_type=env_config.dataset_type,
        )

        instances = generator.generate(args.num_instances, device=device)

        print(f"\nEvaluating on synthetic {args.dataset_type} "
              f"({args.num_jobs}x{args.num_machines}), "
              f"{args.num_instances} instances")

        t0 = time.time()
        if args.strategy == 'greedy':
            makespan = evaluate_greedy(policy, env, instances, device)
        else:
            makespan = evaluate_sampling(policy, env, instances, args.num_samples, device)
        elapsed = time.time() - t0

        print(f"Strategy: {args.strategy}")
        print(f"Average makespan: {makespan.mean().item():.2f}")
        print(f"Std makespan: {makespan.std().item():.2f}")
        print(f"Time: {elapsed:.2f}s")

    elif args.mode == 'benchmark':
        if args.benchmark_dir is None:
            raise ValueError("--benchmark_dir required for benchmark mode")

        benchmark_files = sorted([
            f for f in os.listdir(args.benchmark_dir)
            if f.endswith('.fjs') or f.endswith('.txt')
        ])

        print(f"\nEvaluating on {len(benchmark_files)} benchmark instances")
        print(f"Strategy: {args.strategy}")

        results = []
        for fname in benchmark_files:
            fpath = os.path.join(args.benchmark_dir, fname)
            try:
                instance = load_benchmark_instance(fpath, device)

                t0 = time.time()
                if args.strategy == 'greedy':
                    makespan = evaluate_greedy(policy, env, instance, device)
                else:
                    makespan = evaluate_sampling(policy, env, instance, args.num_samples, device)
                elapsed = time.time() - t0

                ms = makespan.item()
                results.append(ms)
                print(f"  {fname}: makespan = {ms:.1f}, time = {elapsed:.2f}s")
            except Exception as e:
                print(f"  {fname}: ERROR - {e}")

        if results:
            print(f"\nAverage makespan: {np.mean(results):.2f}")
            print(f"Std makespan: {np.std(results):.2f}")


if __name__ == '__main__':
    main()
