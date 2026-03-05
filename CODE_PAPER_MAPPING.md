# ReSched 代码与论文对照说明

> 论文：*ReSched: Rethinking Flexible Job Shop Scheduling from a Transformer-based Architecture with Simplified States* (ICLR 2026 Under Review)

---

## 目录

1. [整体架构对照](#1-整体架构对照)
2. [状态表示（Section 4.1）](#2-状态表示section-41)
3. [Operation Branch + RoPE（Section 4.2.1）](#3-operation-branch--ropesection-421)
4. [Machine Branch + Edge-in-Attention（Eq. 5）](#4-machine-branch--edge-in-attentioneq-5)
5. [Self-based Cross-attention（Eq. 6）](#5-self-based-cross-attentioneq-6)
6. [Decision-Making MLP（Section 4.2.2）](#6-decision-making-mlpsection-422)
7. [REINFORCE 训练算法（Algorithm 1）](#7-reinforce-训练算法algorithm-1)
8. [MDP 与奖励设计（Eq. 3）](#8-mdp-与奖励设计eq-3)
9. [实例生成与数据集类型](#9-实例生成与数据集类型)
10. [超参数对照](#10-超参数对照)

---

## 1. 整体架构对照

论文 Figure 2 描述了 ReSched 的总体框架，分为两个模块：

```
状态 (State)
    │
    ├─── Feature Extraction（特征提取）
    │        ├─── Operation Branch（Self-Attention + RoPE）
    │        └─── Machine Branch（Cross-Attention + Edge-in-Attn + Self-based Cross-Attn）
    │
    └─── Decision-Making（MLP 打分）
             └─── Softmax → 概率分布 → 选择 (operation, machine) 对
```

代码中的对应关系：

| 论文模块 | 代码类 | 文件 |
|---------|--------|------|
| Feature Extraction | `FeatureExtraction` | `model/network.py` |
| Operation Branch | `OperationBranch` | `model/network.py` |
| Machine Branch | `MachineBranch` | `model/network.py` |
| Decision-Making | `DecisionMaking` | `model/network.py` |
| 完整策略网络 | `ReSchedPolicy` | `model/network.py` |
| RoPE | `RotaryPositionalEmbedding` | `model/rope.py` |
| FJSP 环境 (MDP) | `FJSPEnv` | `env/fjsp_env.py` |
| 实例生成器 | `InstanceGenerator` | `env/instance_generator.py` |
| 训练脚本 | `train_epoch`, `main` | `train.py` |

---

## 2. 状态表示（Section 4.1）

### 2.1 四个核心特征（Section 4.1.2 "State: Features"）

论文定义了 4 个核心特征：

| 特征 | 类型 | 论文来源 |
|------|------|--------|
| Operation Available Time（相对值） | 操作节点特征 | Definition 4.1, (1) |
| Machine Available Time（相对值） | 机器节点特征 | Definition 4.1, (2) |
| Duration | 边特征（O2M 连接上） | Definition 4.1, (4) |
| Minimum Duration | 操作节点特征 | Section 4.1.2 "State: Features" |

代码中的对应（`env/fjsp_env.py` → `_get_state()`，`model/network.py` → `FeatureExtraction.forward()`）：

```python
# fjsp_env.py: _get_state()
rel_op_available  = self.op_available_time  - batch_min_time   # 操作可用时间（相对值）
rel_machine_available = self.machine_available_time - batch_min_time  # 机器可用时间（相对值）
# duration_matrix 直接来自实例（边特征）
# min_duration 在 InstanceGenerator 中预计算

# network.py: FeatureExtraction.forward()
op_features = torch.stack([op_avail, min_dur], dim=-1)  # (B, N, 2) → op_embed → (B, N, D)
m_features  = m_avail.unsqueeze(-1)                     # (B, M, 1) → machine_embed → (B, M, D)
edge_features = dur_matrix.unsqueeze(-1)                # (B, N, M, 1) → edge_embed → (B, N, M, D)
```

### 2.2 相对时间归一化（Section 4.1.2 "Relative Available Time"）

> 论文："We normalize all operation and machine available times by subtracting the global minimum available time at each step."

```python
# fjsp_env.py: _get_state()
batch_min_time[b] = torch.min(eligible_op_times.min(), machine_available_time[b].min())
rel_op_available      = self.op_available_time      - batch_min_time.unsqueeze(1)
rel_machine_available = self.machine_available_time - batch_min_time.unsqueeze(1)
```

### 2.3 O2O 向后跳跃连接（Section 4.1.2 "O2O Connection"）

> 论文："we adopt backward-looking edges... we introduce hop connections from each operation to **all its successors**"

含义：操作 $O_{i,j}$ 的 O2O 连接指向 job $i$ 中所有 **后继** 操作 $O_{i,j+1}, O_{i,j+2}, \ldots$（而不仅仅是直接后继），这样无需多层消息传递就能获取全局 job 约束。

```python
# instance_generator.py: _generate_single()
for i in range(n_ops):
    for j in range(i + 1, n_ops):    # op i → 所有后继 op j（跳跃连接）
        o2o_mask[start + i, start + j] = True
```

操作一旦被调度，就从图中移除：

```python
# fjsp_env.py: step()
self.o2o_mask[b, op_idx, :] = False   # 移除该操作的所有出边
self.o2o_mask[b, :, op_idx] = False   # 移除所有指向该操作的入边
self.o2m_mask[b, op_idx, :] = False   # 移除其 O2M 连接
```

### 2.4 动作空间与约束（Section 4.1.2 "Action"）

> 论文："The only restriction we impose is the natural precedence constraint between operations."

代码中 `action_mask` 仅由 precedence（`eligible` 掩码）和机器兼容性（`o2m_mask`）决定，不使用任何启发式剪枝：

```python
# fjsp_env.py: _get_state()
action_mask = self.eligible.unsqueeze(-1) & self.o2m_mask   # (B, max_ops, M)
```

`eligible` 表示"job 内第一个未调度的操作"，即满足 precedence 约束的操作。

---

## 3. Operation Branch + RoPE（Section 4.2.1）

### 3.1 论文核心方程（Eq. 4）

$$\langle \text{RoPE}_q(x_a, a),\ \text{RoPE}_k(x_b, b)\rangle = g(x_a, x_b, a - b)$$

注意力权重是内容 $x_a, x_b$ 与 **相对位置差** $a-b$ 的函数。此处 $a, b$ 为 **intra-job 位置索引**（同 job 内的操作序号）。

### 3.2 代码实现（`model/network.py` → `OperationBranch`）

```python
# OperationBranch.forward()
q = self.q_proj(x_norm).view(B, N, H, d_k).transpose(1, 2)  # (B, H, N, d_k)
k = self.k_proj(x_norm).view(B, N, H, d_k).transpose(1, 2)
v = self.v_proj(x_norm).view(B, N, H, d_k).transpose(1, 2)

q, k = self.rope(q, k, positions)   # 对 Q/K 施加 RoPE，positions = op_positions（intra-job 位置）

scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
scores = scores.masked_fill(~attn_mask, float('-inf'))   # 仅允许关注自身及后继（O2O 掩码）
```

### 3.3 RoPE 实现（`model/rope.py`）

```python
# 旋转公式：[x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
rot_x1 = x1 * cos - x2 * sin
rot_x2 = x1 * sin + x2 * cos
```

- **无额外可学习参数**（与论文"without introducing additional learnable parameters"一致）
- `positions` 使用 `op_positions`（intra-job 位置），而非全局操作序号
- RoPE **仅在 Operation Branch** 中使用，Machine Branch 不使用位置编码（与论文一致）

### 3.4 O2O 掩码在注意力中的作用

每个操作只能关注自身和其 job 内的后继操作（由 `o2o_attn_mask` 控制）：

```python
# fjsp_env.py: _get_state()
o2o_attn_mask = self.o2o_mask.clone()        # 后继连接
# 加上自环（每个未调度操作关注自身）
self_mask = diag & unscheduled.unsqueeze(1) & unscheduled.unsqueeze(2)
o2o_attn_mask = o2o_attn_mask | self_mask
```

---

## 4. Machine Branch + Edge-in-Attention（Eq. 5）

### 4.1 论文核心方程（Eq. 5）

$$\text{Attention}(M_m, O_{ij}) = \sigma\!\left(\frac{(q_m + q_{m,ij})^\top (k_{ij} + k_{m,ij})}{\sqrt{d}}\right) \cdot (v_{ij} + v_{m,ij})$$

其中 $q_{m,ij},\ k_{m,ij},\ v_{m,ij}$ 是从边特征（加工时长 $D^m_{ij}$）投影出的边特定向量，**Q/K/V 以及所有 head 共享同一套投影权重**。

### 4.2 代码实现（`model/network.py` → `MachineBranch`）

```python
# 单一共享边投影（论文"share projection weights across heads and Q/K/V"）
self.edge_proj = nn.Linear(hidden_dim, hidden_dim)

# forward() 中：
edge_proj = self.edge_proj(edge_emb)                  # (B, N, M, D) — 共享投影
edge_for_attn = edge_proj.permute(0,1,3,2,4)          # (B, H, M, N, d_k)

q_total = q_m_exp + edge_for_attn                     # q_m + q_{m,ij}
k_total = k_o_exp + edge_for_attn                     # k_{ij} + k_{m,ij}
v_total = v_o_exp + edge_for_attn                     # v_{ij} + v_{m,ij}

scores = (q_total * k_total).sum(dim=-1) / math.sqrt(d_k)   # (B, H, M, N)
```

边信息同时影响 Q、K、V，使得边特征不仅改变注意力权重，也影响最终聚合表示（与论文"influence not only the attention weights but also the final aggregated representations"一致）。

---

## 5. Self-based Cross-attention（Eq. 6）

### 5.1 论文核心方程（Eq. 6）

$$h'_m = \alpha_{mm} v_m + \sum_{(ij)\in\mathcal{N}(M_m)} \alpha_{ij} v_{ij}$$

机器节点 $M_m$ 对自身分配一个 **软注意力权重** $\alpha_{mm}$，与所有连接操作的权重一起经过 softmax 归一化。这解决了操作数量远多于机器数量（通常 10:1）导致的注意力稀释问题。

### 5.2 代码实现（`model/network.py` → `MachineBranch`）

**修复后**，机器自注意力使用独立的 K/V 投影（与操作的 K/V 投影参数隔离）：

```python
# __init__ 中：
self.q_proj = nn.Linear(hidden_dim, hidden_dim)    # 机器 Query（cross-attention）
self.k_proj = nn.Linear(hidden_dim, hidden_dim)    # 操作 Key
self.v_proj = nn.Linear(hidden_dim, hidden_dim)    # 操作 Value
self.k_self_proj = nn.Linear(hidden_dim, hidden_dim)  # 机器自注意力 Key（独立）
self.v_self_proj = nn.Linear(hidden_dim, hidden_dim)  # 机器自注意力 Value（独立）

# forward() 中：
# 计算机器对自身的注意力分数
k_self = self.k_self_proj(machine_norm)...         # 独立 Key 投影
self_scores = (q_m * k_self).sum(dim=-1)           # (B, H, M)

# 将 self-score 与 cross-score 拼接后整体 softmax（实现 Eq. 6 的软权重分配）
combined_scores = torch.cat([self_scores.unsqueeze(-1), scores], dim=-1)   # (B, H, M, N+1)
combined_mask   = torch.cat([self_mask_flag, cross_mask], dim=-1)
attn_weights = F.softmax(combined_scores.masked_fill(~combined_mask, -inf), dim=-1)

# 独立 Value 投影用于自身聚合
v_self = self.v_self_proj(machine_norm)...
combined_values = torch.cat([v_self.unsqueeze(3), v_total], dim=3)         # (B, H, M, N+1, d_k)
attn_out = einsum('bhmi,bhmid->bhmd', attn_weights, combined_values)
```

> **修复说明**：原实现复用了操作的 `k_proj` 和 `v_proj` 来计算机器自注意力，导致机器自注意力与操作 cross-attention 共享参数空间，语义上不正确。修复后为机器自注意力添加了独立的 `k_self_proj` 和 `v_self_proj`。

---

## 6. Decision-Making MLP（Section 4.2.2）

### 6.1 论文设计

> "It consists of a multi-layer perceptron (MLP) that takes as input the operation and machine embeddings... along with the edge (duration) embeddings, and produces a scalar score for each feasible pair. A softmax over these scores yields the final probability distribution."

### 6.2 代码实现（`model/network.py` → `DecisionMaking`）

```python
# 拼接三类嵌入（维度 3D）
op_exp  = op_emb.unsqueeze(2).expand(B, N, M, D)      # (B, N, M, D)
m_exp   = machine_emb.unsqueeze(1).expand(B, N, M, D) # (B, N, M, D)
pair_features = torch.cat([op_exp, m_exp, edge_emb], dim=-1)  # (B, N, M, 3D)

# MLP: (3D → 64 → ... → 1)，输出标量分数
scores = self.mlp(pair_features).squeeze(-1)           # (B, N, M)

# 非法动作填充 -inf，然后 log_softmax
scores = scores.masked_fill(~action_mask, float('-inf'))
log_probs = F.log_softmax(scores.view(B, -1), dim=-1)  # (B, N*M)
```

默认 MLP 结构：`3*128 → 64 → 64 → 1`（3层，与论文"3-layer MLP"一致）。

---

## 7. REINFORCE 训练算法（Algorithm 1）

### 7.1 论文 Algorithm 1

```
For each epoch:
  1. 生成一批实例
  2. 展开策略（rollout），获取轨迹 (s_t, a_t, r_t)
  3. 计算折扣回报 R_t = Σ_{t'≥t} γ^{t'-t} r_{t'}
  4. 归一化优势 A_t = (R_t - mean(R_t)) / std(R_t)
  5. 策略梯度更新：∇ -Σ_t A_t · log π(a_t|s_t)
```

### 7.2 代码实现（`train.py`）

```python
# rollout()：展开策略
while not env.done.all():
    actions, log_probs = policy.select_action(state, greedy=False)
    next_state, rewards, done = env.step(actions)
    log_probs_list.append(log_probs)
    rewards_list.append(rewards)

# compute_returns()：折扣回报（从后向前累积）
for t in reversed(range(T)):
    running_return = rewards_list[t] + gamma * running_return
    returns[t] = running_return

# train_epoch()：归一化 + 策略梯度
advantages = returns - returns.mean(dim=1, keepdim=True)
advantages = advantages / std.clamp(min=1e-8)

log_probs  = torch.stack(log_probs_list, dim=0)    # (T, B)
policy_loss = -(advantages.detach() * log_probs).mean()
```

---

## 8. MDP 与奖励设计（Eq. 3）

### 8.1 奖励函数（Eq. 3）

$$r_t = -\left(\overline{FT}_{\max}(s_{t+1}) - \overline{FT}_{\max}(s_t)\right)$$

即每一步的奖励为**估计下界 makespan 的负增量**（越小越好）。$\overline{FT}_{\max}$ 通过对所有未调度操作按最小加工时长迭代计算得到。

### 8.2 代码实现（`env/fjsp_env.py`）

```python
# _compute_lb_makespan()：估计下界
for job j:
    for each op in job j (in order):
        if scheduled: prev_ft = actual_finish_time
        else:         prev_ft = prev_ft + min_duration   # 使用最小加工时长估计
lb[b] = max(lb[b], prev_ft)

# step() 中的奖励计算
old_lb = self.lb_makespan.clone()
# ... 执行动作 ...
new_lb = self._compute_lb_makespan()
reward = -(new_lb - old_lb)    # 对应 Eq. 3
```

### 8.3 状态转移（Eq. 2）

$$FT_{ij} = \max(FT_{i(j-1)},\ AT^m_t) + D^m_{ij}$$

```python
# step() 中：
start_time  = max(op_avail, m_avail)          # max(FT_{i(j-1)}, AT^m_t)
finish_time = start_time + duration           # + D^m_{ij}
self.op_finish_time[b, op_idx]      = finish_time
self.machine_available_time[b, m_idx] = finish_time   # 更新机器可用时间
```

---

## 9. 实例生成与数据集类型

论文使用两种随机数据集生成方式（Section 5）：

| 数据集 | ops/job 范围 | duration 范围 | 代码 (`config.py`) |
|--------|-------------|--------------|-------------------|
| SD1 | $[0.8m, 1.2m]$（$m$ = 机器数） | $[1, 20]$ | `dataset_type='SD1'` |
| SD2 | $[1, m]$ | $[1, 99]$ | `dataset_type='SD2'` |
| JSSP | $m$（固定）| $[1, 99]$ | `dataset_type='JSSP'` |
| FFSP | 3（固定阶段）| $[2, 9]$ | `dataset_type='FFSP'` |

JSSP 退化：每个操作恰好分配给一台机器（机器号为 job 内的随机排列），此时 `o2m_mask` 每行只有一个 True。

---

## 10. 超参数对照

| 超参数 | 论文值 | 代码默认值 | 位置 |
|--------|--------|-----------|------|
| Hidden dim $D$ | 128 | `hidden_dim=128` | `ModelConfig` |
| FFN dim | 512 | `ffn_dim=512` | `ModelConfig` |
| Attention heads | 8 | `num_heads=8` | `ModelConfig` |
| Transformer layers | 2 | `num_layers=2` | `ModelConfig` |
| MLP hidden dim | 64 | `mlp_hidden_dim=64` | `ModelConfig` |
| MLP layers | 3 | `mlp_num_layers=3` | `ModelConfig` |
| Learning rate | $5 \times 10^{-5}$ | `lr=5e-5` | `TrainConfig` |
| Batch size | 50 | `batch_size=50` | `TrainConfig` |
| Epochs | 2000 | `num_epochs=2000` | `TrainConfig` |
| Instances/Epoch | 1000 | `instances_per_epoch=1000` | `TrainConfig` |
| Discount $\gamma$ | 0.99 | `gamma=0.99` | `TrainConfig` |
| Dropout | 0 | `dropout=0.0` | `ModelConfig` |

---

## 总结

| 组件 | 符合度 | 说明 |
|------|:------:|------|
| 4 个核心特征 + 相对时间归一化 | ✅ | Section 4.1.2 完全对应 |
| O2O 向后跳跃连接 | ✅ | 操作可见所有后继，不依赖多层传播 |
| 无启发式动作剪枝 | ✅ | 仅用 precedence + machine eligibility |
| Operation Branch Self-Attention + RoPE | ✅ | Eq. 4，intra-job 位置，无额外参数 |
| Machine Branch Edge-in-Attention | ✅ | Eq. 5，Q/K/V + edge，共享投影 |
| Self-based Cross-attention（修复后） | ✅ | Eq. 6，独立 K/V 投影，软权重分配 |
| Decision-Making MLP（3D 输入，3层） | ✅ | Section 4.2.2 完全对应 |
| REINFORCE + 折扣回报 + 优势归一化 | ✅ | Algorithm 1 完全对应 |
| 奖励 = 负 LB makespan 增量 | ✅ | Eq. 3 完全对应 |
| 超参数 | ✅ | 全部与论文表格一致 |
