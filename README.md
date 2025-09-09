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

### GQA组注意力机制

把多个头划分成若干组（group），**每组内的头共享同一份 K/V**，但不同组之间的 K/V 还是独立的。

#### 手撕代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, head_num: int, kv_head_num: int):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        
        assert self.dim % head_num == 0
        assert head_num % kv_head_num == 0, "head_num must be divisible by kv_head_num"
        
        self.head_dim = self.dim // self.head_num
        self.group_size = self.head_num // self.kv_head_num  # 每组的 query head 数量
        
        # Q 有 head_num 个 head
        self.q = nn.Linear(dim, dim)
        # K 和 V 有 kv_head_num 个 head
        self.k = nn.Linear(dim, kv_head_num * self.head_dim)
        self.v = nn.Linear(dim, kv_head_num * self.head_dim)
        
        self.fc = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        b, s, d = x.shape
        
        # Q: (batch, head_num, seq_len, head_dim)
        q = self.q(x).view(b, s, self.head_num, self.head_dim).transpose(1, 2)
        
        # K, V: (batch, kv_head_num, seq_len, head_dim)
        k = self.k(x).view(b, s, self.kv_head_num, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, s, self.kv_head_num, self.head_dim).transpose(1, 2)
        
        # 将 K 和 V 重复以匹配 Q 的 head 数量
        # 每个 KV head 对应 group_size 个 Q head
        k = k.repeat_interleave(self.group_size, dim=1)  # (batch, head_num, seq_len, head_dim)
        v = v.repeat_interleave(self.group_size, dim=1)  # (batch, head_num, seq_len, head_dim)
        
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

## RLHF（**基于人类反馈的强化学习**）

### PPO

PPO代表近端策略优化，它需要以下组件：

1、策略（Actor、Policy）模型（$\pi_{\theta}$）：经过预训练或SFT后的模型，该模型是可学习的，需要打开训练。

2、约束模型（$\pi_{ref}$）：该模型是冻结的，不需要参与训练，用来做KL散度约束，用来防止Actor模型的分布偏离原始模型太远。

3、奖励模型（$R_{\phi}$）：一个经过训练并冻结的模型，它为给定 prompt 的**完整响应**提供奖励分值，有时不需要，基于规则也可以给奖励。

4、价值模型（$V_{\Phi}$）：它是一个可学习的模型，它为给定 prompt 的**部分响应**提供奖励分值。

![img](https://p.ipic.vip/wqdjwt.jpg)

PPO的整个流程包含以下六个部分：

1、**生成响应**：LLM为给定单个prompt生成单个响应；

2、**响应打分**：奖励模型为每个响应分配奖励；

3、**响应价值**：价值模型给出多步响应对应的价值

4、**计算优势**（**advantages**）：使用 **GAE**（广义优势估计）计算优势；

5、**优化策略（policy）：**通过优化总目标来更新 LLM；

6、**更新价值函数（critic）**：训练价值函数，使其更好地预测给定部分响应的奖励。

#### 广义优势估计（GAE）

我们的策略通过优化**优势函数（advantage function）**来更新。直观地讲，优势函数定义了在特定状态$s_{t}$（即提示 prompt + 到目前为止生成的词）下**选择某个动作**$a_{t}$（即词）**相较于平均动作的优越程度**。优势函数的定义如下：
$$
A_{t} = Q(s_{t}, a_{t}) - V(s_{t})
$$
$Q(s_{t}, a_{t}))$: **动作价值（action value）**，表示在状态$s_{t}$下选择动作$a_{t}$的期望回报；

$V(s_{t})$: **状态价值（state value）**，表示状态$s_{t}$下的期望回报。

为了估计这个优势函数$A_t$，通常有两种主要方法，每种方法都有其优缺点：

- **蒙特卡洛方法（Monte-Carlo, MC）**：

- - **方法**：使用完整轨迹的奖励（即完整响应的奖励）来估计优势。
  - **优点**：这种方法具有**低偏差（low bias）**，因为它能准确地反映完整轨迹的真实奖励。
  - **缺点**：由于**奖励稀疏**，需要采样足够多的完整轨迹才能进行优化。这会导致**高方差（high variance）**，并且代价昂贵。

- **时序差分方法（Temporal Difference, TD）**：

- - **方法**：使用一步轨迹奖励（即刚生成的 token 的奖励）来估计优势。
  - **优点**：这种方法能在 token 级别计算奖励，显著减少了方差（low variance）。
  - **缺点**：由于无法准确预测完整响应的最终奖励，这种方法会引入较高的偏差（high bias）。

- 为了在偏差和方差之间取得平衡，引入了 General Advantage Estimation (GAE)。GAE 通过多步 TD 来实现这一目标。

TD误差（步长为1）：$\delta_{t} = R_{t+1} + \gamma V_{\Phi}(s_{t+1}) - V_{\Phi}(s_{t})$

GAE（步长为n）：
$$
\begin{align*}
A_{t}^{n} &= R_{t+1} + \gamma^2 R_{t+2} + \dots + \gamma^n V_{\Phi}(s_{t+n}) - V_{\Phi}(s_{t}) \\
          &= \sum_{l=1}^{n} \gamma^{l-1} \delta_{t+l-1}
\end{align*}
$$
最终形式：$A_{t}^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^{l} \delta_{t+l}$

PPO 的主要目标是最大化 advantage（$A_{t}^{\text{GAE}}$ ） 地方，也就是说，我们希望让 LLM 生成的每一个 token 都能够最大化 reward（或者说，最大化 advantage——即每个 token 要比它“平均水平”的表现更好）。这个目标通过以下公式实现：
$$
\mathcal{L}_{\text{clip}}(\theta) = \mathbb{E}_{t}\left[\min\left(c_{t}(\pi_{\theta})A_{t}^{\text{GAE}}, \operatorname{clip}(c_{t}(\pi_{\theta}), 1 - \epsilon, 1 + \epsilon)A_{t}^{\text{GAE}}\right)\right]
$$
$c_{t}(\pi_{\theta}) = \frac{\pi_{\theta}(a_{t} | s_{t})}{\pi_{\theta_{\text{old}}}(a_{t} | s_{t})}$: 是在给定累计状态 $s_t$下，策略更新前后的概率比值；

$\epsilon$: 是一个超参数，用于控制 clip 范围；

$A_{t}^{\text{GAE}}$: 是之前通过 GAE 计算得到的优势值。

#### Critic loss

理论上，我们并没有办法获得真实的折扣回报，因此无法获取 Critic 的明确标签。而在 Actor-Critic 范式中使用包含了真实奖励的折扣回报$\hat{G_t}$ 作为标签。下面是 Critic loss 的计算公式：
$$
\mathcal{L}(\Phi) = \mathbb{E}_{t}\left[\max\left(\left(V_{\Phi}(s_{t}) - \hat{G}_{t}\right)^2, \left(\operatorname{clip}\left(V_{\Phi}(s_{t}), V_{\Phi}^{\text{old}}(s_{t}) - \epsilon, V_{\Phi}^{\text{old}}(s_{t}) + \epsilon\right) - \hat{G}_{t}\right)^2\right)\right]
$$

#### KL散度

除了最大化 advantage，PPO 还引入了 KL 惩罚，防止当前策略偏离我们微调前的原始模型太远：
$$
\text{KL}(\theta) = \mathbb{E}_{s_t}\left[\mathbb{D}_{\text{KL}}\left(\pi_{\theta}(\cdot|s_t) \| \pi_{\text{ref}}(\cdot|s_t)\right)\right]
$$

$$
\mathbb{D}_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
$$

在计算 KL 散度时需要计算两个模型在整个词表上的概率，为了简化计算过程，在实现时通常采用近似计算。如下图所示

![img](https://p.ipic.vip/9cvd0e.jpg)

#### 熵奖励（Entropy Bonus）

熵奖励用于鼓励 LLM 探索更多的输出，而不是一味选择概率最高的词：$H(\theta) = -\mathbb{E}_{a_i \sim \pi_{\theta}}\left[\log \pi_{\theta}(a_i | s)\right]$

PPO最终的损失loss：
![image-20250909222233990](https://p.ipic.vip/emjd2u.png)

### GRPO

GRPO 相较于 PPO 的主要改进为：

- 使用对同一问题的多个采样输出的平均奖励作为基线**；**
- 优势计算中去掉了对值函数$V_{\Phi}(s_{t})$的依赖**。**

下面是 GRPO 计算 advantage 的具体流程：

- **采样多个响应**：对每个 prompt，采样一组 response：$r = \{r_{1}, r_{2}, \dots, r_{G}\}$

- **通过 Reward Model 打分**：对每个响应$r_i$，使用奖励模型$R_{\phi}(r_{i})$得到一个分数。对应的产生$G$个奖励分值为$r = \{R_{1}, R_{2}, \dots, R_{G}\}$
- **用 group 内标准化估计 Advantage**：$A_{i,t} = R_{i} = \frac{R_{\phi}(r_{i}) - \text{mean}(R)}{\text{std}(R)}$

下面是 GRPO 的具体流程，对比图PPO 的流程，二者的主要区别在于 advantage 的计算。

![img](https://p.ipic.vip/9owr8k.jpg)

### DAPO

DAPO 的出发点非常直接：在实际训练中，GRPO 往往因 clip 范围设置不合理、采样冗余以及长序列梯度被稀释等问题，导致大量训练信号被浪费。针对这些问题，DAPO 逐一提出改进，形成了四个核心优化点。
$$
\begin{align*}
\mathcal{J}_{\text{DAPO}}(\theta) = \; & \mathbb{E}_{\substack{(q,a) \sim P(Q) \\ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)}} \left[ \frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \right. \\
& \left. \min\left(r_{i,t}(\theta)A_i, \operatorname{clip}(r_{i,t}(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}})A_i\right) \right] \\
& \text{s.t.,} \quad 0 < \left| \{o_i \mid \text{is\_equivalent}(a, o_i)\} \right| < G
\end{align*}
$$


#### 为什么 DAPO 提高了 $ 1 + \epsilon_{high}$  的上界？

作者发现，如果 clip 的上界$\epsilon$设置过小，会出现这样的问题：当 old policy 对某个 token 的概率很低，而该 token 的 advantage 又是正值（即 old model 恰好采样得非常好），此时当前 policy model 的上涨空间就会受到很大限制，而上涨恰恰是我们希望发生的。

举例来说，如果 old policy 的概率是 0.9， $\epsilon = 0.2$，clip 上界为$0.9 \times 1.2 = 1.08$ ，已超过概率的最大值 1.0，这种情况是绝对不会被 clip 的；但如果 old policy 的概率是 0.2，clip 上界仅为 0.24，即便当前模型将其概率提升到 0.4（一个不是非常激进且恰到好处的改进），也会因$\epsilon$过小而被 clip，导致该 token 的训练信号被废弃。为了解决这一问题，DAPO 引入 **Clip-Higher**，提高上界以提升 token 利用效率。

这类似于“马太效应”——*富人越来越富，穷人很难翻身*。如果 old policy 难得采到一个关键 token（例如 `「Wait」`）且概率极低，而当前模型对此 token 的概率提升显著，却因为 clip 限制过紧被抹掉，那么模型几乎没有翻盘的机会。

Clip-Higher 解决了“好 token 涨幅受限”的问题，但并未触及另一个浪费来源——采样多样性不足。为此，DAPO 引入了 **动态采样**。

#### DAPO - 动态采样

DAPO 的第二个创新是 **动态采样**（Dynamic Sampling）。这项技术的背景是：假如一个 query 我们 sample 了 10 次，这 10 次每次都答得很好/或者很差，都取得了 max reward/zero reward，这个时候由于 GRPO 的计算方法，导致这 10 次采样的 advantage 都是 0，所以这些采样所带来的 gradient 就也都是 0；这样做的一个后果就是，实际的有梯度的 sample 要远低于名义 sample 数，导致最后梯度汇集的时候没有收集到足够的信息，从而形成高方差、不稳定的训练，以及 sample 的浪费。需要注意的是，这种现象是在训练初期；以及后期随着训练的进行在不断加强的，因为刚开始时模型效果很差，而训练越到后边模型效果越好，给出满分回答的几率就越大。因此，DAPO 在采集样本时，额外做了一件事：保证每次采样出来的回答，reward 不全是 0 或者 1，如果采样出来的回答全是 0 或者 1 就继续采样，直到不满足为止。这也是损失函数中$\text{s.t.,} \quad 0 < \left| \{o_i \mid \text{is\_equivalent}(a, o_i)\} \right| < G$的来源，它保证同一输入下的采样集合中既包含正确回答，也包含错误回答。

除了多样性问题，GRPO 在长回答训练中还有一个隐性缺陷——**token 梯度权重随回答长度增加而被稀释**。DAPO 的第三个改进正是 **Token-Level Gradient Loss**。

#### DAPO - Token-Level Gradient Loss

DAPO 第三个方面的创新是为了解决 GRPO 在训练长回答时 gradient 的权重会随着采样回答的长度变长而下降的问题。首先解释为什么采样长度变长权重会下降。假设采样了 2 次，有一次回答一共有 200 个 token，而另一次回答有 10 个 token。那么根据 GRPO 的计算公式，每次回答的梯度先在 sample 内求平均，再在 batch 内求平均。第一次回答每个 token 的权重是$(1/200) * (1/2)$，而第二个回答每个 token 的权重是$(1/10) * (1/2)$，所以第二次回答的 token 的影响要明显高于第一次回答。再来说采样长度变长权重下降的危害：对于一些比较难的问题，长回答本身就很正常，如果这些回答本身非常好，那么由于长度平均就会导致本来非常有用的梯度信号被稀释；假如回答是不好的，长度长仅仅也是因为重复单词，或者回答冗余词太多，长度归一就导致这次采样本该带来的纠正信号没法正确传递到 policy model 上。总结来说就是：

这会带来两个问题：

1. **长高质量回答**的有用信号被稀释；
2. **长低质量回答**的纠正信号也被稀释（长只是因为冗余或重复）。

所以 DAPO 采用的方法是：把一次梯度计算时所有采样生成的 token 总数加起来求平均，回到上边这个例子，第一次采样和第二次采样每个 token 的权重都是 $(1/200) * (1/2)$，即对不同回答中的 token 一视同仁。这样就能改善 GRPO 在长样本训练中的效率低下的问题。这对应着损失函数中的改变，公式上，对于 loss 的 aggregation 方式由原来 GRPO 的$\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}$改为$\frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|}$

实验证明，token-level loss 不仅训练更稳定，还能有效控制 entropy：过高会导致策略趋于随机，过低则探索不足（Clip-Higher 可缓解该问题）。通过将 sample-level loss 改为 token-level loss，DAPO 让长回答能够按比例影响最终梯度，每个 token 的损失都直接参与整体更新。

最后一个改进同样与回答长度相关，但关注点不同——它处理的是**过长回答对整体奖励的负面影响**。

#### DAPO - Overlong Reward Shaping

DAPO 的第四个改进是在奖励设计中引入 **软惩罚机制**（Soft Punishment）来处理过长回答。具体来说，当生成长度超过第一个预设阈值时，惩罚会随长度线性增加；一旦超过第二个阈值，惩罚将抵消因回答正确获得的所有奖励，相当于将该回答视为无效。这种惩罚是按 token 作用在 reward（即 advantage）上的。

综上，DAPO 在 **Clip-Higher、动态采样、Token-Level Gradient Loss** 和 **Overlong Reward Shaping** 四个方面，对 GRPO 进行了精细化改造，显著提升了训练的效率与稳定性。不过在某些特定架构（尤其是 MoE）下，GRPO 的结构性问题依然存在，这就引出了下一节的 **GSPO**。

### GSPO
