"""
FJSP Environment for ReSched.

Implements the MDP formulation:
  - State: Operation/Machine Available Time (relative), Duration, Min Duration,
           O2O backward hop connections, O2M connections.
  - Action: Select an (operation, machine) pair.
  - Transition: Update finish times and available times.
  - Reward: Negative difference of estimated lower-bound makespan.
"""
import torch
from typing import Dict, Tuple, Optional


class FJSPEnv:
    """
    Batched FJSP environment. All operations are done on tensors for GPU efficiency.
    """

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device

    def reset(self, instance: Dict) -> Dict:
        """
        Reset environment with a new batch of instances.

        Args:
            instance: Dict from InstanceGenerator.generate()

        Returns:
            state: Dict with current state representation.
        """
        self.batch_size = instance['duration_matrix'].shape[0]
        self.max_ops = instance['duration_matrix'].shape[1]
        self.num_machines = instance['num_machines']

        # Instance data (immutable)
        self.duration_matrix = instance['duration_matrix']                # (B, max_ops, M)
        self.job_indices = instance['job_indices']                        # (B, max_ops)
        self.op_positions = instance['op_positions']                      # (B, max_ops)
        self.op_mask = instance['op_mask'].clone()                        # (B, max_ops) True if op exists
        self.o2m_mask_init = instance['o2m_mask'].clone()                 # (B, max_ops, M)
        self.o2o_mask_init = instance['o2o_mask'].clone()                 # (B, max_ops, max_ops)
        self.min_duration = instance['min_duration']                      # (B, max_ops)
        self.num_ops_per_job = instance['num_ops_per_job']                # (B, num_jobs)
        self.total_ops = instance['total_ops']                            # (B,)

        # Scheduling state
        self.op_available_time = torch.zeros(
            self.batch_size, self.max_ops, device=self.device
        )
        self.machine_available_time = torch.zeros(
            self.batch_size, self.num_machines, device=self.device
        )
        self.op_finish_time = torch.zeros(
            self.batch_size, self.max_ops, device=self.device
        )

        # Current O2O and O2M masks (evolve as operations are scheduled)
        self.o2o_mask = self.o2o_mask_init.clone()
        self.o2m_mask = self.o2m_mask_init.clone()

        # Track which operations are scheduled
        self.scheduled = torch.zeros(
            self.batch_size, self.max_ops, dtype=torch.bool, device=self.device
        )
        # Track first unscheduled operation per job (precedence constraint)
        # An operation is eligible if it's the first unscheduled op in its job
        self.eligible = torch.zeros(
            self.batch_size, self.max_ops, dtype=torch.bool, device=self.device
        )
        self.step_count = 0
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # Cache identity matrix for _get_state (reused every step)
        self._diag = torch.eye(
            self.max_ops, dtype=torch.bool, device=self.device
        ).unsqueeze(0)  # (1, max_ops, max_ops)

        # Cache job start indices for vectorised helpers (must come before eligible update)
        self._refresh_job_starts()

        self._update_eligible_batch()

        # Compute initial lower-bound makespan
        self.lb_makespan = self._compute_lb_makespan()

        return self._get_state()

    def _refresh_job_starts(self):
        """Recompute job_starts cache (call once per reset since ops don't change)."""
        B = self.batch_size
        num_jobs = self.num_ops_per_job.shape[1]
        self._job_starts = torch.zeros(
            B, num_jobs, dtype=torch.long, device=self.device
        )
        if num_jobs > 1:
            self._job_starts[:, 1:] = self.num_ops_per_job[:, :-1].cumsum(dim=1)
        self._max_ops_per_job = int(self.num_ops_per_job.max().item())

    def _update_eligible(self):
        """Alias – delegate to vectorised version."""
        self._update_eligible_batch()

    def _update_eligible_batch(self):
        """
        Vectorised update of eligible operations.
        Iterates over *positions within a job* (≤ max_ops_per_job, typically small)
        instead of over the full B × J × N product.
        """
        self.eligible.fill_(False)
        job_starts = self._job_starts          # (B, num_jobs)
        max_pos   = self._max_ops_per_job

        # found[b, j] = True once job j in batch b has its eligible op set
        found = torch.zeros(
            self.batch_size, self.num_ops_per_job.shape[1],
            dtype=torch.bool, device=self.device
        )

        for pos in range(max_pos):
            # Which (batch, job) pairs have an op at this position and are not done?
            has_pos = (pos < self.num_ops_per_job) & ~self.done.unsqueeze(1)  # (B, J)
            op_idx  = (job_starts + pos).clamp(0, self.max_ops - 1)          # (B, J)

            # Is the op at this position unscheduled?
            not_sched = ~self.scheduled.gather(1, op_idx)  # (B, J)

            # Set eligible only if: valid, unscheduled, and no earlier op was chosen
            should_set = has_pos & not_sched & ~found
            found = found | should_set

            if should_set.any():
                b_idx, j_idx = should_set.nonzero(as_tuple=True)
                self.eligible[b_idx, op_idx[b_idx, j_idx]] = True

    def step(self, actions: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Execute actions for the batch (vectorised implementation).

        Args:
            actions: (B, 2) tensor where actions[:, 0] = operation index,
                     actions[:, 1] = machine index.

        Returns:
            state: Dict with next state.
            reward: (B,) reward tensor.
            done: (B,) done flags.
        """
        op_indices = actions[:, 0]   # (B,)
        m_indices  = actions[:, 1]   # (B,)
        B = self.batch_size
        batch_range = torch.arange(B, device=self.device)
        active = ~self.done          # (B,)

        old_lb = self.lb_makespan.clone()

        # ── duration, start, finish ─────────────────────────────────────────
        duration  = self.duration_matrix[batch_range, op_indices, m_indices]  # (B,)
        op_avail  = self.op_available_time[batch_range, op_indices]            # (B,)
        m_avail   = self.machine_available_time[batch_range, m_indices]        # (B,)
        finish_time = torch.max(op_avail, m_avail) + duration                 # (B,)

        # ── update state tensors (only for active instances) ─────────────────
        old_op_ft = self.op_finish_time[batch_range, op_indices]
        self.op_finish_time[batch_range, op_indices] = torch.where(
            active, finish_time, old_op_ft
        )

        old_m_avail = self.machine_available_time[batch_range, m_indices]
        self.machine_available_time[batch_range, m_indices] = torch.where(
            active, finish_time, old_m_avail
        )

        # mark scheduled
        self.scheduled[batch_range, op_indices] = (
            self.scheduled[batch_range, op_indices] | active
        )

        # ── O2O / O2M mask: clear the scheduled op (loop over batch, O(B)) ──
        for b in range(B):
            if not active[b]:
                continue
            oi = op_indices[b].item()
            self.o2o_mask[b, oi, :] = False
            self.o2o_mask[b, :, oi] = False
            self.o2m_mask[b, oi, :]  = False

        # ── update successor available time (vectorised) ─────────────────────
        op_pos = self.op_positions[batch_range, op_indices]  # (B,)
        job_id = self.job_indices[batch_range, op_indices]   # (B,)
        succ_mask = (
            (self.job_indices == job_id.unsqueeze(1))         # same job
            & (self.op_positions == (op_pos + 1).unsqueeze(1))  # next position
            & self.op_mask                                     # valid op
        )  # (B, max_ops)
        update_mask = succ_mask & active.unsqueeze(1)          # (B, max_ops)
        self.op_available_time = torch.where(
            update_mask,
            finish_time.unsqueeze(1).expand_as(self.op_available_time),
            self.op_available_time,
        )

        self.step_count += 1
        self._update_eligible_batch()

        # done once every valid op is scheduled
        self.done = (self.scheduled | ~self.op_mask).all(dim=1)

        # ── reward ───────────────────────────────────────────────────────────
        new_lb = self._compute_lb_makespan()
        reward = -(new_lb - old_lb)
        self.lb_makespan = new_lb

        return self._get_state(), reward, self.done.clone()

    def _compute_lb_makespan(self) -> torch.Tensor:
        """
        Vectorised lower-bound makespan.
        Iterates over positions within a job (≤ max_ops_per_job) rather than
        over the full B × J × N product.
        """
        B         = self.batch_size
        job_starts = self._job_starts      # (B, J)
        max_pos    = self._max_ops_per_job

        # Accumulated per-job finish-time lower bound
        job_lb_ft = torch.zeros(
            B, self.num_ops_per_job.shape[1], device=self.device
        )

        for pos in range(max_pos):
            has_pos  = pos < self.num_ops_per_job                           # (B, J)
            op_idx   = (job_starts + pos).clamp(0, self.max_ops - 1)       # (B, J)
            is_valid = has_pos & self.op_mask.gather(1, op_idx)             # (B, J)

            is_sched  = self.scheduled.gather(1, op_idx)                    # (B, J)
            actual_ft = self.op_finish_time.gather(1, op_idx)               # (B, J)
            min_dur   = self.min_duration.gather(1, op_idx)                 # (B, J)

            # LB for this op: carry forward from prev or use actual finish
            lb_ft  = job_lb_ft + min_dur
            new_ft = torch.where(is_sched, actual_ft, lb_ft)
            job_lb_ft = torch.where(is_valid, new_ft, job_lb_ft)

        return job_lb_ft.max(dim=1).values  # (B,)

    def get_makespan(self) -> torch.Tensor:
        """Get the actual makespan for each instance (vectorised)."""
        ft = self.op_finish_time.masked_fill(~self.op_mask, 0.0)
        return ft.max(dim=1).values

    def _get_state(self) -> Dict:
        """
        Construct current state representation.

        The state has:
          - Relative available times (subtract global min)
          - Duration (edge feature)
          - Minimum duration
          - O2O mask (backward hop connections)
          - O2M mask (connections)
          - Eligible mask (which ops are eligible for scheduling)
          - Action mask (which op-machine pairs are feasible)
        """
        # Compute global minimum available time per instance (vectorised)
        # Use a large sentinel for non-eligible ops so min ignores them
        _BIG = 1e9
        masked_op_times = self.op_available_time.masked_fill(~self.eligible, _BIG)
        min_op = masked_op_times.min(dim=1).values          # (B,)
        min_m  = self.machine_available_time.min(dim=1).values  # (B,)
        batch_min_time = torch.min(min_op, min_m)
        # Clamp artefacts from all-masked rows and zero out done instances
        batch_min_time = batch_min_time.clamp(max=_BIG - 1)
        batch_min_time = batch_min_time * (~self.done).float()
        batch_min_time = batch_min_time.masked_fill(batch_min_time >= _BIG, 0.0)

        # Relative available times
        rel_op_available = self.op_available_time - batch_min_time.unsqueeze(1)
        rel_machine_available = self.machine_available_time - batch_min_time.unsqueeze(1)

        # Build action mask: (B, max_ops, M) - True if (op, machine) is a valid action
        action_mask = self.eligible.unsqueeze(-1) & self.o2m_mask  # (B, max_ops, M)

        # O2O attention mask — reuse cached identity matrix
        o2o_attn_mask = self.o2o_mask.clone()  # (B, max_ops, max_ops)
        diag = self._diag.expand(self.batch_size, -1, -1)  # (B, N, N) — no copy
        unscheduled = (~self.scheduled) & self.op_mask  # (B, max_ops)
        self_mask = diag & unscheduled.unsqueeze(1) & unscheduled.unsqueeze(2)
        o2o_attn_mask = o2o_attn_mask | self_mask

        return {
            'op_available_time': rel_op_available,          # (B, max_ops)
            'machine_available_time': rel_machine_available,  # (B, M)
            'duration_matrix': self.duration_matrix,          # (B, max_ops, M)
            'min_duration': self.min_duration,                # (B, max_ops)
            'o2o_mask': o2o_attn_mask,                       # (B, max_ops, max_ops)
            'o2m_mask': self.o2m_mask,                        # (B, max_ops, M)
            'op_mask': self.op_mask & (~self.scheduled),      # (B, max_ops)
            'eligible': self.eligible,                        # (B, max_ops)
            'action_mask': action_mask,                       # (B, max_ops, M)
            'op_positions': self.op_positions,                 # (B, max_ops)
            'job_indices': self.job_indices,                   # (B, max_ops)
            'scheduled': self.scheduled,                       # (B, max_ops)
        }

    def get_action_space_size(self, state: Dict) -> torch.Tensor:
        """Return the number of feasible actions per instance."""
        return state['action_mask'].sum(dim=(1, 2))
