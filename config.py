"""
Configuration for ReSched.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class EnvConfig:
    """Environment configuration."""
    # Problem size
    num_jobs: int = 10
    num_machines: int = 5
    # Number of operations per job: sampled from [ops_low, ops_high]
    ops_low: Optional[int] = None   # default: int(0.8 * num_machines) for SD1
    ops_high: Optional[int] = None  # default: int(1.2 * num_machines) for SD1
    # Duration range
    duration_low: int = 1
    duration_high: int = 20  # SD1: [1, 20], SD2: [1, 99]
    # Dataset type: 'SD1' or 'SD2'
    dataset_type: str = 'SD1'

    def __post_init__(self):
        if self.dataset_type == 'SD1':
            if self.ops_low is None:
                self.ops_low = max(1, int(0.8 * self.num_machines))
            if self.ops_high is None:
                self.ops_high = int(1.2 * self.num_machines)
            self.duration_low = 1
            self.duration_high = 20
        elif self.dataset_type == 'SD2':
            if self.ops_low is None:
                self.ops_low = 1
            if self.ops_high is None:
                self.ops_high = self.num_machines
            self.duration_low = 1
            self.duration_high = 99
        elif self.dataset_type == 'JSSP':
            if self.ops_low is None:
                self.ops_low = self.num_machines
            if self.ops_high is None:
                self.ops_high = self.num_machines
            self.duration_low = 1
            self.duration_high = 99
        elif self.dataset_type == 'FFSP':
            if self.ops_low is None:
                self.ops_low = 3  # 3 stages
            if self.ops_high is None:
                self.ops_high = 3
            self.duration_low = 2
            self.duration_high = 9


@dataclass
class ModelConfig:
    """Model configuration."""
    hidden_dim: int = 128
    ffn_dim: int = 512
    num_heads: int = 8
    num_layers: int = 2
    mlp_hidden_dim: int = 64
    mlp_num_layers: int = 3
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Training configuration."""
    # REINFORCE
    lr: float = 5e-5
    gamma: float = 0.99
    num_epochs: int = 2000
    instances_per_epoch: int = 1000
    batch_size: int = 50
    # Validation
    val_size: int = 100
    val_freq: int = 10  # validate every N epochs
    # Sampling strategy for evaluation
    num_samples: int = 100
    # Seed
    seed: int = 42
    # Device
    device: str = 'cuda'
    # Save
    save_dir: str = 'checkpoints'
    # Logging
    log_freq: int = 10
    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class Config:
    """Full configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
