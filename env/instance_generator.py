"""
Instance generator for FJSP, JSSP, and FFSP.
Generates batched instances as tensors for efficient GPU-based training.
"""
import torch
import numpy as np
from typing import Tuple, Dict


class InstanceGenerator:
    """
    Generates random FJSP instances in a batched manner.

    An FJSP instance is defined by:
      - num_jobs: number of jobs
      - num_machines: number of machines
      - ops_per_job: number of operations per job (can vary)
      - duration_matrix: (num_total_ops, num_machines) with 0 for infeasible
      - job_indices: mapping from operation to job
      - op_positions: position of each operation within its job (for RoPE)
    """

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        ops_low: int,
        ops_high: int,
        duration_low: int = 1,
        duration_high: int = 20,
        dataset_type: str = 'SD1',
    ):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.ops_low = ops_low
        self.ops_high = ops_high
        self.duration_low = duration_low
        self.duration_high = duration_high
        self.dataset_type = dataset_type

    def generate(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Dict:
        """
        Generate a batch of FJSP instances.

        Returns a dict:
          - 'duration_matrix': (batch, max_ops, num_machines) float, 0 if infeasible
          - 'job_indices': (batch, max_ops) int, job index for each op
          - 'op_positions': (batch, max_ops) int, position within job
          - 'op_mask': (batch, max_ops) bool, True if op exists
          - 'o2m_mask': (batch, max_ops, num_machines) bool, True if op-machine pair feasible
          - 'o2o_mask': (batch, max_ops, max_ops) bool, O2O backward hop connections
          - 'num_ops_per_job': (batch, num_jobs) int
          - 'min_duration': (batch, max_ops) float
        """
        batch_data = []
        for _ in range(batch_size):
            batch_data.append(self._generate_single())

        return self._collate(batch_data, device)

    def _generate_single(self) -> Dict:
        """Generate a single FJSP instance."""
        num_jobs = self.num_jobs
        num_machines = self.num_machines

        # Sample number of operations per job
        ops_per_job = np.random.randint(self.ops_low, self.ops_high + 1, size=num_jobs)
        total_ops = int(ops_per_job.sum())

        # Duration matrix: (total_ops, num_machines)
        duration_matrix = np.zeros((total_ops, num_machines), dtype=np.float32)

        # O2M feasibility
        o2m_mask = np.zeros((total_ops, num_machines), dtype=bool)

        op_idx = 0
        job_indices = []
        op_positions = []

        for job_id in range(num_jobs):
            n_ops = ops_per_job[job_id]
            for pos in range(n_ops):
                if self.dataset_type == 'JSSP':
                    # JSSP: each op goes to exactly one machine
                    # Assign machines as a permutation for each job
                    pass
                else:
                    # FJSP: randomly assign machines
                    # Each op can be processed by a random subset of machines (at least 1)
                    num_eligible = np.random.randint(1, num_machines + 1)
                    eligible_machines = np.random.choice(
                        num_machines, size=num_eligible, replace=False
                    )
                    for m in eligible_machines:
                        dur = np.random.randint(self.duration_low, self.duration_high + 1)
                        duration_matrix[op_idx, m] = dur
                        o2m_mask[op_idx, m] = True

                job_indices.append(job_id)
                op_positions.append(pos)
                op_idx += 1

        # For JSSP: each job has ops_per_job ops, each assigned to a unique machine
        if self.dataset_type == 'JSSP':
            op_idx = 0
            for job_id in range(num_jobs):
                n_ops = ops_per_job[job_id]
                machine_perm = np.random.permutation(num_machines)[:n_ops]
                for pos in range(n_ops):
                    m = machine_perm[pos]
                    dur = np.random.randint(self.duration_low, self.duration_high + 1)
                    duration_matrix[op_idx, m] = dur
                    o2m_mask[op_idx, m] = True
                    op_idx += 1

        job_indices = np.array(job_indices, dtype=np.int64)
        op_positions = np.array(op_positions, dtype=np.int64)

        # Minimum duration for each operation (across feasible machines)
        min_duration = np.zeros(total_ops, dtype=np.float32)
        for i in range(total_ops):
            feasible_durations = duration_matrix[i, o2m_mask[i]]
            if len(feasible_durations) > 0:
                min_duration[i] = feasible_durations.min()

        # O2O mask: backward hop connections (each op connects to all its successors in the same job)
        o2o_mask = np.zeros((total_ops, total_ops), dtype=bool)
        op_idx = 0
        for job_id in range(num_jobs):
            n_ops = ops_per_job[job_id]
            start = op_idx
            for i in range(n_ops):
                for j in range(i + 1, n_ops):
                    # op i can see op j (successor) — backward-looking from successor
                    o2o_mask[start + i, start + j] = True
            op_idx += n_ops

        return {
            'duration_matrix': duration_matrix,
            'job_indices': job_indices,
            'op_positions': op_positions,
            'o2m_mask': o2m_mask,
            'o2o_mask': o2o_mask,
            'ops_per_job': ops_per_job,
            'min_duration': min_duration,
            'total_ops': total_ops,
        }

    def _collate(self, batch_data: list, device: torch.device) -> Dict:
        """Collate a list of instances into batched tensors with padding."""
        batch_size = len(batch_data)
        max_ops = max(d['total_ops'] for d in batch_data)

        duration_matrix = torch.zeros(batch_size, max_ops, self.num_machines, device=device)
        job_indices = torch.zeros(batch_size, max_ops, dtype=torch.long, device=device)
        op_positions = torch.zeros(batch_size, max_ops, dtype=torch.long, device=device)
        op_mask = torch.zeros(batch_size, max_ops, dtype=torch.bool, device=device)
        o2m_mask = torch.zeros(batch_size, max_ops, self.num_machines, dtype=torch.bool, device=device)
        o2o_mask = torch.zeros(batch_size, max_ops, max_ops, dtype=torch.bool, device=device)
        min_duration = torch.zeros(batch_size, max_ops, device=device)
        num_ops_per_job = torch.zeros(batch_size, self.num_jobs, dtype=torch.long, device=device)
        total_ops = torch.zeros(batch_size, dtype=torch.long, device=device)

        for b, d in enumerate(batch_data):
            n = d['total_ops']
            duration_matrix[b, :n] = torch.from_numpy(d['duration_matrix'])
            job_indices[b, :n] = torch.from_numpy(d['job_indices'])
            op_positions[b, :n] = torch.from_numpy(d['op_positions'])
            op_mask[b, :n] = True
            o2m_mask[b, :n] = torch.from_numpy(d['o2m_mask'])
            o2o_mask[b, :n, :n] = torch.from_numpy(d['o2o_mask'])
            min_duration[b, :n] = torch.from_numpy(d['min_duration'])
            num_ops_per_job[b] = torch.from_numpy(d['ops_per_job'])
            total_ops[b] = n

        return {
            'duration_matrix': duration_matrix,
            'job_indices': job_indices,
            'op_positions': op_positions,
            'op_mask': op_mask,
            'o2m_mask': o2m_mask,
            'o2o_mask': o2o_mask,
            'min_duration': min_duration,
            'num_ops_per_job': num_ops_per_job,
            'total_ops': total_ops,
            'num_machines': self.num_machines,
        }
