# 25.12.16汇报 Qwen3-Next 模型架构

## 1. 整体架构
- **总层数**: 48层 = 12 × (3层Gated DeltaNet + 1层Gated Attention)的重复结构
- **隐藏维度**: 2048

## 2. Gated DeltaNet 结构

### 2.1 结构信息
| 组件 | 头数 | 头维度 | 总维度 |
|------|------|--------|--------|
| Q/K | 16 | 128 | 2048 (16×128) |
| V   | 32 | 128 | 4096 (32×128) |

### 2.2 线性投影
**输入**: `X` (形状: `[B, L, 2048]`)

**投影矩阵**: 
- `W_qkvz` (形状: `[2048, 12288]`)
- `12288 = k_dim(2048) + v_dim(4096) + z_dim(4096) + q_dim(2048)`
- `W_ba` (形状: `[2048, 64]`)

**投影操作**: 
```python
X @ W_qkvz = [q, k, v, z]  # 总维度 12288
X @ W_ba = [b, a]  # 总维度 64
```

**维度分解**:
- `q`: `[B, L, 32, 128]` (query)
- `k`: `[B, L, 16, 128]` (key)
- `v`: `[B, L, 16, 128]` (value)
- `z`: `[B, L, 32, 128]` (门控参数，与qkv投影一起得到)
- `b`: `[B, L, 32]` (β参数的原始值)
- `a`: `[B, L, 32]` (衰减函数的原始值)

### 2.3 卷积变换
对特征通道进行卷积变换：
- `q`: `[B, L, 32, 128]` → `[B, L, 4096]`
- `k`: `[B, L, 16, 128]` → `[B, L, 2048]`
- `v`: `[B, L, 16, 128]` → `[B, L, 2048]`

### 2.4 激活函数
使用 **SiLU激活函数**

### 2.5 Gated Delta Rule 线性注意力

#### 步骤1: 重复q和k头以匹配v头数
```python
Q_repeat = repeat(q, [1, 1, 2, 1])  # [B, L, 32, 128]
K_repeat = repeat(k, [1, 1, 2, 1])  # [B, L, 32, 128]
```

#### 步骤2: DeltaNet状态更新（递归形式）
```python
# 初始化状态
S = zeros(B, 32, 128, 128)  # [B, 32, 128, 128]

# 计算β和g
β = torch.sigmoid(b) # (B, L, 32)
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias) # (B, L, 32)

for t = 1 to L:
    # 获取当前时间步的各个组件
    Q_t = Q_repeat[:, t, :, :]  # [B, 32, 128]
    K_t = K_repeat[:, t, :, :]  # [B, 32, 128]
    V_t = V[:, t, :, :]         # [B, 32, 128]
    β_t = β[:, t, :]            # [B, 32]
    g_t = exp(g[:, t, :])       # [B, 32]
    
    # 扩展维度用于广播
    g_t_exp = expand(g_t, [-1, -1, 128, 128])  # [B, 32, 128, 128]
    β_t_exp = expand(β_t, [-1, -1, 128])       # [B, 32, 128]
    
    # 状态更新
    S = S * g_t_exp                     # 衰减
    ΔS = K_t^T @ (V_t * β_t_exp)        # 外积更新
    S = S + ΔS
    
    # 计算输出
    O_t = Q_t @ S  # [B, 32, 128]
```

#### 步骤3: 合并所有时间步
```python
O = stack([O_t for t in 1..L], dim=1)  # [B, L, 32, 128]
```

### 对比标准注意力方法
- `标准注意力`  ：O(B * L^2 * d), KV缓存
- `Gated DeltaNet线性注意力` ：O(B * L * d^2), 存储状态S

## 3. Gated Attention 结构
> 进行补充

## 4. MoE
### MoE整体架构概览

Qwen3Next的MoE模块采用**稀疏激活的混合专家系统**，主要特点：
- **专家数量**：512个专家
- **激活专家数**：每个token激活10个专家（Top-10）
- **共享专家**：1个共享专家，总是激活
- **专家中间维度**：512（MoE专家），共享专家有自己的中间维度
  
**关键配置**：
- `num_experts`: 512
- `num_experts_per_tok`: 10（每个token激活的专家数）
- `moe_intermediate_size`: 512（专家中间维度）
- `shared_expert_intermediate_size`: 共享专家的中间维度（通常更大）

### 🔄 MoE模块的矩阵运算流程

#### 1. **输入准备**
```
输入: hidden_states ∈ ℝ^(B×L×H)  # B=batch, L=seq_len, H=hidden_size=2048

# 重塑为2D矩阵
hidden_states_reshaped = hidden_states.view(-1, H)  # (B*L, 2048)
```

#### 2. **门控网络（Router）计算**

##### **门控权重矩阵**
```python
# weight: (512, 2048)
# hidden_states: (B*L, 2048)
router_logits = F.linear(hidden_states, weight)  # (B*L, 512)
```

#### **矩阵运算分解**
```
W_router ∈ ℝ^(512×2048)  # 路由器权重
X ∈ ℝ^(B*L×2048)         # 展平的输入

# 线性变换，得到的结果可看作每个token和512个专家之间的关系
router_logits = X @ W_router^T  # (B*L×2048) @ (2048×512) → (B*L×512)

# Softmax计算（分专家维度）
router_probs = softmax(router_logits, dim=-1)  # (B*L×512)

# Top-K选择
router_top_value, router_indices = topk(router_probs, k=10, dim=-1)
# router_top_value: (B*L×10), router_indices: (B*L×10)

### 3. **共享专家计算**

#### **共享专家MLP结构**
```python
class Qwen3NextMLP:
    def __init__(self):
        self.gate_proj = nn.Linear(2048, shared_intermediate_size)
        self.up_proj = nn.Linear(2048, shared_intermediate_size)
        self.down_proj = nn.Linear(shared_intermediate_size, 2048)
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

#### **矩阵运算**
```
输入: X ∈ ℝ^(B*L×2048)

# Gate投影
W_gate_shared ∈ ℝ^(shared_intermediate_size×2048)
gate_shared = X @ W_gate_shared^T  # (B*L×shared_intermediate_size)

# Up投影
W_up_shared ∈ ℝ^(shared_intermediate_size×2048)
up_shared = X @ W_up_shared^T  # (B*L×shared_intermediate_size)

# 激活和逐元素乘
hidden_shared = SiLU(gate_shared) * up_shared  # (B*L×shared_intermediate_size)

# Down投影
W_down_shared ∈ ℝ^(2048×shared_intermediate_size)
shared_expert_output = hidden_shared @ W_down_shared^T  # (B*L×2048)

# 共享专家门控
W_shared_gate ∈ ℝ^(1×2048)
shared_gate = sigmoid(X @ W_shared_gate^T)  # (B*L×1)
shared_expert_output = shared_expert_output * shared_gate  # 广播乘法
```

### 4. **稀疏专家计算（核心）**

#### **专家参数存储方式**
```python
class Qwen3NextExperts(nn.Module):
    def __init__(self, config):
        # 3D参数存储
        self.gate_up_proj = nn.Parameter(torch.empty(
            self.num_experts, 2 * self.intermediate_dim, self.hidden_dim
        ))  # (512, 1024, 2048)
        
        self.down_proj = nn.Parameter(torch.empty(
            self.num_experts, self.hidden_dim, self.intermediate_dim
        ))  # (512, 2048, 512)
```

#### **稀疏计算流程**

**步骤1：创建专家掩码**
```python
# router_indices: (B*L, 10) - 每个token选中的10个专家索引
expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=512)
# expert_mask: (B*L, 10, 512)

# 转置以便按专家处理
expert_mask = expert_mask.permute(2, 1, 0)  # (512, 10, B*L)

# 找出哪些专家至少被一个token选中
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
# expert_hit: (活跃专家数, 1)
```

**步骤2：稀疏循环计算**
```python
final_hidden_states = torch.zeros_like(hidden_states_reshaped)  # (B*L, 2048)

for expert_idx in expert_hit:
    expert_idx = expert_idx[0]  # 当前专家索引
    
    # 找到选中该专家的所有token
    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
    # top_k_pos: 在该token的top-k中的位置 (0-9)
    # token_idx: token索引
    
    current_state = hidden_states_reshaped[token_idx]  # (num_tokens, 2048)
    
    # Gate-Up投影
    gate_up = current_state @ self.gate_up_proj[expert_idx].T  # (num_tokens, 1024)
    gate, up = gate_up.chunk(2, dim=-1)  # 各(num_tokens, 512)
    
    # 激活和逐元素乘
    current_hidden_states = self.act_fn(gate) * up  # (num_tokens, 512)
    
    # Down投影
    current_hidden_states = current_hidden_states @ self.down_proj[expert_idx].T  # (num_tokens, 2048)
    
    # 应用路由权重
    current_weights = router_top_value[token_idx, top_k_pos, None]  # (num_tokens, 1)
    current_hidden_states = current_hidden_states * current_weights
    
    # 累加到最终输出
    final_hidden_states.index_add_(0, token_idx, current_hidden_states)
```

#### **矩阵运算的等价形式**

虽然实际实现是稀疏循环，但从数学上可以表示为：

```
# 定义稀疏张量运算
对于每个专家e：
    mask_e = expert_mask[e]  # (10, B*L)
    对于mask_e中的每个非零位置(k, t)：
        权重 = router_top_value[t, k]
        输入_t = hidden_states_reshaped[t]  # (2048)
        
        # 专家e的计算
        输出_e_t = SiLU(输入_t @ W_gate_up_e[:512]^T) * (输入_t @ W_gate_up_e[512:]^T)
        输出_e_t = 输出_e_t @ W_down_e^T  # (2048)
        
        # 加权累加
        final_hidden_states[t] += 权重 * 输出_e_t
```

### 5. **最终输出合并**

```python
# 添加共享专家输出
final_hidden_states += shared_expert_output  # (B*L, 2048)

# 重塑回3D
final_hidden_states = final_hidden_states.view(B, L, 2048)  # (B, L, 2048)
```

## 📈 维度变化总表

| 步骤 | 操作 | 输入维度 | 输出维度 | 说明 |
|------|------|----------|----------|------|
| 1 | 输入重塑 | (B,L,2048) | (B*L,2048) | 展平以便处理 |
| 2 | 路由器计算 | (B*L,2048) | (B*L,512) | 线性投影+softmax |
| 3 | Top-K选择 | (B*L,512) | (B*L,10)×2 | 路由权重和专家索引 |
| 4 | 共享专家计算 | (B*L,2048) | (B*L,2048) | 完整MLP计算 |
| 5 | 专家掩码创建 | (B*L,10) | (512,10,B*L) | one-hot编码+转置 |
| 6 | 稀疏专家计算 | 多个输入 | (B*L,2048) | 循环计算活跃专家 |
| 7 | 合并输出 | (B*L,2048)×2 | (B*L,2048) | 加和 |
| 8 | 输出重塑 | (B*L,2048) | (B,L,2048) | 恢复序列维度 |

### MoE的核心设计

#### 1. **稀疏激活机制**
- 每个token只激活10/512个专家（~2%）
- 大幅降低计算量，同时保持模型容量

#### 2. **3D参数存储**
```
gate_up_proj: (512, 1024, 2048)  # 所有专家的gate+up参数
down_proj: (512, 2048, 512)      # 所有专家的down参数
```
- 便于批量索引和计算
- 内存连续，访问高效

#### 3. **负载均衡损失**

在训练时添加辅助损失函数，确保专家使用均衡：

```python
def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask):
    # gate_logits: 各层的路由器logits元组
    # 计算每个专家被选中的概率
    # 计算每个专家的token分配比例
    # 损失 = num_experts * Σ(概率 * 分配比例)
```

**数学公式**：
```
对于每个MoE层：
    P = softmax(router_logits)  # (B*L, E)
    A = topk_mask(P)  # (B*L, E)，选中为1否则为0
    
    # 专家使用频率
    expert_usage = mean(A, dim=0)  # (E,)
    # 路由器概率
    router_prob = mean(P, dim=0)   # (E,)
    
    # 负载均衡损失
    loss_lb = num_experts * sum(expert_usage * router_prob)
```

#### 4. **共享专家设计**
- 总是激活，提供基础处理能力
- 门控机制控制共享专家的贡献程度
- 确保即使稀疏路由有问题，也有稳定输出

### 计算复杂度分析

### 密集计算（理论上）：
```
每个token的专家计算：
512个专家 × (2048×512 + 512×2048) = 512 × 2.1M ≈ 1.07B 操作/token
```

#### 稀疏计算（实际）：
```
每个token的专家计算：
10个专家 × (2048×512 + 512×2048) = 10 × 2.1M ≈ 21M 操作/token
```

**计算减少**：约51倍（512/10）

#### 内存访问模式：
```
# 密集计算：需要加载所有512个专家的参数
内存访问量: 512 × (1024+512)×2048 × 4字节 ≈ 6.3GB

# 稀疏计算：只加载10个活跃专家的参数
内存访问量: 10 × (1024+512)×2048 × 4字节 ≈ 123MB
```

