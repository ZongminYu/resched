# ReSched: Rethinking Flexible Job Shop Scheduling from a Transformer-based Architecture with Simplified States

PyTorch 复现论文 "ReSched: Rethinking Flexible Job Shop Scheduling from a Transformer-based Architecture with Simplified States" (ICLR 2026 Under Review).

## 项目结构

```
resched/
├── __init__.py
├── config.py                 # 配置类 (环境、模型、训练)
├── utils.py                  # 工具函数
├── train.py                  # REINFORCE 训练脚本
├── evaluate.py               # 评估脚本 (greedy / sampling)
├── env/
│   ├── __init__.py
│   ├── fjsp_env.py           # FJSP 环境 (MDP 实现)
│   └── instance_generator.py # 随机实例生成器
└── model/
    ├── __init__.py
    ├── rope.py               # Rotary Positional Embedding
    └── network.py            # 策略网络 (双分支 Transformer + MLP)
```

## 核心设计

### 状态表示 (4个特征)
1. **Operation Available Time** — 操作可用时间（相对值）
2. **Machine Available Time** — 机器可用时间（相对值）
3. **Duration** — 加工时长（边特征）
4. **Minimum Duration** — 最小加工时长

### 网络架构
- **Operation Branch**: Self-attention + RoPE（编码job内操作的相对位置）
- **Machine Branch**: Cross-attention + Edge-in-Attention + Self-based Cross-attention
- **Decision Making**: 3层 MLP，对每个可行 (operation, machine) 对打分

### 训练算法
- **REINFORCE** (Algorithm 1 in paper)
- Reward: 估计下界 makespan 的差分
- Discount factor γ = 0.99

## 快速开始

### 训练

```bash
# 在 SD1 数据集上训练 10×5 规模
python -m resched.train \
    --num_jobs 10 \
    --num_machines 5 \
    --dataset_type SD1 \
    --num_epochs 2000 \
    --batch_size 50 \
    --lr 5e-5

# 在 SD2 数据集上训练
python -m resched.train \
    --num_jobs 10 \
    --num_machines 5 \
    --dataset_type SD2 \
    --num_epochs 2000 \
    --batch_size 50

# JSSP 训练
python train.py \
    --num_jobs 10 \
    --num_machines 10 \
    --dataset_type JSSP

# 小规模快速测试
python -m resched.train \
    --num_jobs 5 \
    --num_machines 3 \
    --dataset_type SD1 \
    --num_epochs 10 \
    --instances_per_epoch 100 \
    --batch_size 10
```

### 评估

```bash
# Greedy 评估
python -m resched.evaluate \
    --checkpoint checkpoints/resched_SD1_10x5_best.pt \
    --mode synthetic \
    --num_jobs 10 \
    --num_machines 5 \
    --strategy greedy

# Sampling 评估 (100 trajectories)
python -m resched.evaluate \
    --checkpoint checkpoints/resched_SD1_10x5_best.pt \
    --mode synthetic \
    --strategy sampling \
    --num_samples 100

# Benchmark 评估
python -m resched.evaluate \
    --checkpoint checkpoints/resched_SD1_10x5_best.pt \
    --mode benchmark \
    --benchmark_dir data/brandimarte/
```

## 超参数 (论文默认)

| 参数 | 值 |
|------|------|
| Hidden dim | 128 |
| FFN dim | 512 |
| Attention heads | 8 |
| Transformer layers | 2 |
| MLP hidden dim | 64 |
| MLP layers | 3 |
| Learning rate | 5×10⁻⁵ |
| Batch size | 50 |
| Epochs | 2000 |
| Instances/Epoch | 1000 |
| Discount γ | 0.99 |

## 支持的问题类型

- **FJSP** (Flexible Job Shop Scheduling Problem) — 主要目标
- **JSSP** (Job Shop Scheduling Problem) — 特殊情况：每个操作固定一台机器
- **FFSP** (Flexible Flow Shop Scheduling Problem) — 特殊情况：共享阶段序列

## 论文关键贡献

1. **极简状态**: 仅 4 个核心特征 + 图结构，去除冗余历史信息
2. **RoPE**: 在 Operation Branch 中编码 job 内操作的相对位置
3. **Edge-in-Attention**: 在 Machine Branch 的 cross-attention 中直接嵌入边特征（加工时长）
4. **Self-based Cross-attention**: 缓解操作-机器数量不平衡导致的注意力稀释
