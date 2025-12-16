# qwen3-next 模型架构
1. 总层数48 -->12*(3*Gated DeltaNet + Gated Attention)
2. 隐藏维度 2048
  一张图
  
## Gated DeltaNet结构
#### 结构信息
QK的头数16    头维度128    16*128=2048
V的头数32     头维度128    32*128=4096

#### 线性投影
#### 单行代码
`print("Hello World")`

#### 多行代码（指定语言）
```输入X            #2048

W_qkvz           # 2048*12288     12288=k_dim(2024)+v_dim(4096)+z_dim(4096)+q_dim(2048)

X@W_qkvz   # 12288=k_dim(2024)+v_dim(4096)+z_dim(4096)+q_dim(2048) 

q:(B,L,32,128)
k:(B,L,16,128)
v:(B,L,16,128)
z:(B,L,32,128)
\```

#### 纯文本代码块


#### 卷积变换
q:(B,L,32,128)  ->(B,L,4096)
k:(B,L,16,128)  ->(B,L,2048)
v:(B,L,16,128)  ->(B,L,2048)

按照特征通道进行卷积变换


#### 激活函数
SiLU激活

### Gated Delta Rule 线性注意力
1. 重复q和k头以匹配v头数 (16 -> 32头)
Q_repeat = repeat(q, [1, 1, 2, 1])  # (B×L×32×128)
K_repeat = repeat(k, [1, 1, 2, 1])  # (B×L×32×128)

2. DeltaNet状态更新（递归形式）
   
#初始化 S = zeros(B, 32, 128, 128)  

for t = 1 to L:
    Q_t = Q_repeat[:, t, :, :]  # (B×32×128)
    K_t = K_repeat[:, t, :, :]  # (B×32×128)
    V_t = V[:, t, :, :]        # (B×32×128)
    β_t = β[:, t, :]           # (B×32)
    g_t = exp(g[:, t, :])      # (B×32)
    
    #扩展维度用于广播
    g_t_exp = expand(g_t, [-1, -1, 128, 128])  # (B×32×128×128)
    β_t_exp = expand(β_t, [-1, -1, 128])       # (B×32×128)
    
    #状态更新
    S = S * g_t_exp  # 衰减
    ΔS = K_t^T @ (V_t * β_t_exp)  # 外积更新
    S = S + ΔS
    
    #计算输出
    O_t = Q_t @ S  # (B×32×128)

4. 合并所有时间步
O = stack([O_t for t in 1..L], dim=1)  # (B×L×32×128)

## Gated Attention结构
