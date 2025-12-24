## 25.12汇报 Qwen3-Next 模型架构

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
![RUNOOB图标](https://img-blog.csdnimg.cn/img_convert/5f192225c466ee7ff1ec1b90363b8c4c.png "RUNOOB",width = 60%)

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

### MoE模块的矩阵运算流程

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

##### **矩阵运算**
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
```

#### MoE的核心设计

#####  **稀疏激活机制**
- 每个token只激活10/512个专家（~2%）
- 大幅降低计算量，同时保持模型容量

#####  **负载均衡损失**

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

#####  **共享专家设计**
- 总是激活，提供基础处理能力
- 门控机制控制共享专家的贡献程度
- 确保即使稀疏路由有问题，也有稳定输出

#### 计算复杂度分析

##### 密集计算（理论上）：
```
每个token的专家计算：
512个专家 × (2048×512 + 512×2048) = 512 × 2.1M ≈ 1.07B 操作/token
```

##### 稀疏计算（实际）：
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

