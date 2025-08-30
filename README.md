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

