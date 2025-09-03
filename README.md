# 大模型八股

## 大模型基础架构

### RoPE旋转位置编码

#### 基础概念

RoPE（Rotary Position Embedding）的核心在于用**旋转矩阵**在**复数域**进行操作替代传统的位置向量叠加操作，既能保留**绝对位置**信息，又能显式建模**相对位置**关系。传统绝对位置编码直接与词向量叠加，经过线性变换后，位置信息的**远程衰减特性**易被破坏。

#### 核心优势

1、远程衰减性：内积结果随相对距离增大呈震荡衰减趋势，符合自然语言中邻近词关联更强的特性。

2、外推能力：旋转操作的周期性允许模型处理超过训练长度的序列，如训练时使用4K长度，推理可扩展至32K。

3、正交性保持：旋转矩阵是正交矩阵，保持向量模长不变，增强模型训练稳定性。

#### 手撕代码

```python
import torch
from typing import Tuple

def precompute_freqs_cis(dim: int, seq_len: int, freqs: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    theta = 1.0 / (freqs ** (torch.arange(0, dim, 2).float() / dim))

    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len)
    # theta.shape = [seq_len, dim // 2] 
    theta = torch.outer(t, theta).float()
    # torch.polar的文档, https://pytorch.org/docs/stable/generated/torch.polar.html
    # torch.polar输入参数是abs和angle，abs所有值都一样，abs和angle的shape都一样
    # torch.polar输入参数是abs和angle，则theta_cis = abs*(cos(angle) + sin(angle)i)
    theta_cis = torch.polar(torch.ones_like(theta), theta)
    return theta_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    theta_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域,  xq_.shape = [batch_size, seq_len, dim // 2]
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * theta_cis).flatten(2) #从dim=2维度开始拍平
    xk_out = torch.view_as_real(xk_ * theta_cis).flatten(2)

    return xq_out.type_as(xq), xk_out.type_as(xk)

if __name__ == '__main__':
    seq_len,dim=3,4
    freqs_cis = precompute_freqs_cis(dim=dim, seq_len=seq_len)
    xq = torch.rand(1, seq_len, dim)
    xk = torch.rand(1, seq_len, dim)
    res = apply_rotary_emb(xq, xk, freqs_cis)
```

### RMSNorm

#### 基础概念

归一化的核心目标是缓解内部协变量偏移问题，即网络中间层输入分布随参数更新而发生改变。主要有以下好处：

1、**提升训练效率**：归一化使每一层的输入分布更加稳定，从而允许使用更高的学习率，加快收敛速度。

2、**保持梯度稳定**：通过控制激活值的尺度，减少梯度消失或者梯度爆炸，适合深层网络的训练。

3、**引入归纳偏置**：归一化在一定程度上对输入进行标准化，降低了模型对输入尺度的敏感性，增强了泛化能力。

#### 核心优势

为什么现在的大模型都用RMSNorm，而不使用BatchNorm和LayerNorm？

1、任务特性不匹配：
	大模型多用于NLP，输入为变长序列，BatchNorm依赖batch统计量，在batch size小和序列长度不固定时不稳定。

​	LayerNorm和RMSNorm按样本独立归一化，天然适配序列建模。

2、训练稳定性：

​	BathNorm在训练和推理阶段行为不同（训练用batch统计量，推理用移动平均），引入不确定性。

​	LayerNorm和RMSNorm训练推理一致，更稳定。

3、可扩展性与效率：

​	BatchNorm在分布式训练中需跨设备同步统计量，通信开销大。

​	LayerNorm和RMSNorm无需跨batch统信，更适合大规模并行训练。

​	RMSNorm进一步简化计算（只有缩放因子gemma，没有中心化beta），减少参数量，提高吞吐。

#### 手撕代码

```python
import torch
from torch import nn

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps) 
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weights
```

### 归一化模块位置的影响

|特征|PostNorm| PreNorm |
| :-:  | :-:  | :-:  |
| 归一化位置 | 残差连接之后 | 残差连接之前 |
|  公式形式  |    y = Norm( x + F(x) )    | y = F( Norm( x ) ) +  x |
|  梯度传播  | 初始阶段梯度较大，后期稳定 | 梯度更平滑，训练更稳定 |
| 收敛速度 | 较慢，需小心调参 | 更快，对学习率更鲁棒 |
| 最终性能 | 可达最优，但训练难度高 | 更易训练，性能更稳定 |

大模型使用PreNorm的优势：
每个子层输入都被归一化，信号尺度更可控。

残差连接保持“恒等映射”主导，梯度传播更平滑。

显著提升训练稳定性，尤其在深层网络中。

### MHA多头注意力机制

#### 基础概念

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

因为这里矩阵的点积，是对矩阵元素逐个相乘再相加，如果序列过长会导致结果过大，除以sqrt（d）是为了进行缩放，防止点积结果过大导致softmax函数梯度消失。简单来说，就是需要压缩softmax输入值，以免输入值过大，进入了softmax的饱和区，导致梯度值太小而难以训练。

#### 核心优势

多个头能够在训练中学会注意到不同的内容。例如在翻译任务中，一些attention head可以关注语法特征，另一些attention head可以关注单词特性。这样模型就可以从不同角度来分析和理解输入信息。

#### 手撕代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim:int, head_num: int):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        assert self.dim % head_num == 0
        self.head_dim = self.dim // self.head_num
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        self.fc = nn.Linear(dim, dim)
    def forward(self, x, mask = None):
        b, s, d = x.shape
        
        q = self.q(x).view(b, s, self.head_num, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, s, self.head_num, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, s, self.head_num, self.head_dim).transpose(1, 2)
        
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
            
        attn_weight = F.softmax(attn_score, dim = -1)
        output = torch.matmul(attn_weight, v)
        
        output = output.transpose(1, 2).contiguous().view(b, s ,d)
        
        return self.fc(output)
```

### 解码中的KV Cache

1、**什么是 KV cache？**

在 Transformer 推理中，把历史生成的 **Key (K)** 和 **Value (V)** 缓存在显存里，供后续新 token 的注意力计算使用。

2、**KV 的优势是什么？**

避免重复计算历史 token 的 K/V，大幅降低推理开销；prefill 阶段后，每一步只需计算新 token 的 Q，再用缓存的 K/V，就能做到 **从 O(n²) 降到 O(n)** 的计算量。

3、**为什么缓存 K/V，而不缓存 Q？**

- Q 只对当前 token 生效，用完即弃，不会在未来复用。

- K/V 会被所有后续 token 的注意力查询重复使用，所以必须缓存。

### MQA多Query注意力机制

#### 基础概念

MQA 是对 **多头注意力（Multi-Head Attention, MHA）** 的一种优化：

**传统 MHA**：每个头都有独立的 Q、K、V 投影矩阵，每个头的 Key/Value 都单独计算并缓存。

**MQA**：每个头仍然有自己的 **Q**（保持不同注意力模式），但 **所有头共享同一份 K/V**。

这样，每个头使用不同的 Q 对同一份 K/V 做注意力计算，显著减少计算量和显存占用。

#### 核心优势

1. **显存占用大幅降低**
   - 推理时需要缓存 KV（KV cache）以加速生成。
   - MHA 每个头都要缓存 K/V，显存占用高；MQA 只缓存一份 K/V，显存降低约为头数倍。
2. **推理速度更快**
   - KV 只计算一次，多头共享，减少重复计算，尤其对长序列生成效果明显。

#### 手撕代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, dim: int, head_num: int):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        assert self.dim % head_num == 0
        self.head_dim = self.dim // self.head_num
        
        # Q 仍然有多个 head
        self.q = nn.Linear(dim, dim)
        # K 和 V 只有一个 head，由所有 query heads 共享
        self.k = nn.Linear(dim, self.head_dim)
        self.v = nn.Linear(dim, self.head_dim)
        
        self.fc = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        b, s, d = x.shape
        
        # Q 有多个 head: (batch, head_num, seq_len, head_dim)
        q = self.q(x).view(b, s, self.head_num, self.head_dim).transpose(1, 2)
        
        # K 和 V 只有一个 head: (batch, seq_len, head_dim)
        k = self.k(x)  # (b, s, head_dim)
        v = self.v(x)  # (b, s, head_dim)
        
        # 将 K 和 V 扩展到所有 head: (batch, head_num, seq_len, head_dim)
        k = k.unsqueeze(1).expand(b, self.head_num, s, self.head_dim)
        v = v.unsqueeze(1).expand(b, self.head_num, s, self.head_dim)
        
        # 计算注意力分数
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
            
        attn_weight = F.softmax(attn_score, dim=-1)
        output = torch.matmul(attn_weight, v)
        
        # 重新组合所有 head 的输出
        output = output.transpose(1, 2).contiguous().view(b, s, d)
        
        return self.fc(output)

```

