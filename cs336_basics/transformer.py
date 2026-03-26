# cs336_basics/transformer.py

import torch
import torch.nn as nn
import math
from einops import einsum


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (..., vocab_size)
    targets: (...,) 整数索引
    """
    # 数值稳定：减去最大值
    logits_max = logits.max(dim=-1, keepdim=True).values
    shifted = logits - logits_max

    # log softmax = shifted[target] - log(sum(exp(shifted)))
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))
    # 取出目标类别的 logit
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = -target_logits + log_sum_exp
    return loss.mean()


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 减最大值保证数值稳定 (softmax 对平移不变)
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
        Q: torch.Tensor, # (..., queries, d_k)
        K: torch.Tensor, # (..., keys, d_k)
        V: torch.Tensor, # (..., keys, d_v)
        mask: torch.Tensor | None = None, # (..., queries, keys) bool
) -> torch.Tensor:  # (..., queries, d_v)
    d_k = Q.shape[-1]

    # 1. 计算注意力分数
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / math.sqrt(d_k)

    # 2. 应用 mask (False 的位置设为 -inf, softmax 后变成 0)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 3. Softmax
    attn_weights = softmax(scores, dim=-1)

    # 4. 加权求和 V
    return einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # 截断正态初始化: std = sqrt(2 / (in + out))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len) -> (batch, seq_len, d_model)
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # 可学习的 gain 参数, 初始化为全 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # 上溯防止溢出

        # RMS(a) = sqrt(mean(a^2) + eps)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.weight

        return result.to(in_dtype)


class _LinearW(nn.Module):
    """SwiGLU 内部专用, 参数名为 weight"""
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        # d_ff 可选: 外部传入或内部计算
        if d_ff is None:
            # d_ff = (8/3) * d_model, 取最近的64的倍数
            d_ff = int(math.ceil(8 * d_model / 3 / 64) * 64)

        self.w1 = _LinearW(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = _LinearW(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = _LinearW(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(W1x) ⊙ W3x
        gate = torch.sigmoid(self.w1(x)) * self.w1(x)   # SiLU = x * sigmoid(x)
        value = self.w3(x)
        return self.w2(gate * value)
    
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # 预计算所有位置的 cos 和 sin, 形状 (max_seq_len, d_k/2)
        # theta_{i, k} = i / Theta^(2k/d)
        k = torch.arange(0, d_k//2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (2 * k / d_k))  # (d_k/2, )
        positions = torch.arange(max_seq_len, device=device).float()    # (max_seq_len, )
        angles = torch.outer(positions, freqs)  # (max_seq_len, d_k/2)

        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, 
                x: torch.Tensor, # (..., seq_len, d_k)
                token_positions: torch.Tensor # (..., seq_len)
                ) -> torch.Tensor:
        # 用 token_positions 取出对应的 cos/sin
        cos = self.cos_cache[token_positions]   # (..., seq_len, d_k/2)
        sin = self.sin_cache[token_positions]   # (..., seq_len, d_k/2)

        # 把 x 的最后一维拆成两两一组
        x_reshaped = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x_even = x_reshaped[..., 0] # (..., seq_len, d_k/2)
        x_odd = x_reshaped[..., 1]  # (..., seq_len, d_k/2)

        # 旋转
        out_even = x_even * cos - x_odd * sin
        out_odd = x_odd * cos + x_even * sin

        # 拼回原来的形状
        out = torch.stack([out_even, out_odd], dim=-1)  # (..., seq_len, d_k/2, 2)
        return out.reshape(*x.shape)
    

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, use_rope: bool = True, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        # 三个投影矩阵 (每个形状都是 d_model × d_model)
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor,  # (batch, seq_len, d_model)
                token_positions: torch.Tensor | None = None,  # (batch, seq_len)
                ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        # 默认 token_positions 是 0, 1, 2, ..., seq_len-1
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)

        # 1. 投影并 reshape 成多头形式
        Q = self.q_proj(x)  # (batch, seq_len, d_model)
        K = self.k_proj(x)  # (batch, seq_len, d_model)
        V = self.v_proj(x)  # (batch, seq_len, d_model)

        # reshape: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 应用 RoPE 到 Q 和 K (head 维度作为 batch 维度)
        # token_positions: (batch, seq_len) -> (batch, 1, seq_len) 广播到所有 head
        # 只在 use_rope=True 时应用 RoPE
        if self.use_rope:
            pos = token_positions.unsqueeze(1).expand(-1, self.num_heads, -1)
            Q = self.rope(Q, pos)  # (batch, num_heads, seq_len, d_k)
            K = self.rope(K, pos)  # (batch, num_heads, seq_len, d_k)

        # 3. 因果 mask: 下三角矩阵, True 表示可以 attend
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))   # (seq_len, seq_len)

        # 4. 注意力 (head 维度自动作为 batch 维度)
        out = scaled_dot_product_attention(Q, K, V, mask=causal_mask)   # out: (batch, num_heads, seq_len, d_k)
        
        # 5. 拼接多头并投影回 d_model 维度
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)    # (batch, seq_len, d_model)

        return self.output_proj(out)
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, use_norm=True, post_norm=False, use_rope=True, ffn_type="swiglu", device=None, dtype=None):
        super().__init__()
        self.use_norm = use_norm
        self.post_norm = post_norm
        if use_norm:
            self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta, use_rope=use_rope, device=device, dtype=dtype)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFFN(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions = None) -> torch.Tensor:
        if not self.use_norm:
            # 第一个子层: 注意力 + 残差
            x = x + self.attn(x, token_positions=token_positions)
            # 第二个子层: FFN + 残差
            x = x + self.ffn(x)
        elif self.post_norm:
            # Post-norm: 残差之后再 norm:
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
        else:
            # Pre-norm (默认)
            # 第一个子层: pre-norm + 注意力 + 残差
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            # 第二个子层: pre-norm + FFN + 残差
            x = x + self.ffn(self.ln2(x))
              
        return x
    

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, 
                 context_length: int,
                 d_model: int, 
                 num_layers: int, 
                 num_heads: int,
                 d_ff: int, 
                 rope_theta: float,
                 use_norm=True,
                 post_norm=False,
                 use_rope=True,
                 ffn_type="swiglu",
                 device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta, use_norm=use_norm, post_norm=post_norm, use_rope=use_rope, ffn_type=ffn_type, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        if use_norm and not post_norm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # self.ln_final = nn.Identity() if post_norm else (RMSNorm(d_model) if use_norm else nn.Identity())
        else:
            self.ln_final = nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        batch, seq_len = token_ids.shape

        # 生成位置序列
        token_positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch, -1)

        # 嵌入
        x = self.token_embeddings(token_ids)  # (batch, seq_len, d_model)

        # 经过所有 Transformer 层
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # 最终归一化 + 输出投影
        x = self.ln_final(x)
        return self.lm_head(x)  # (batch, seq_len, vocab_size)


class SiLUFFN(nn.Module):
    """不带门控的 SiLU 前馈网络, d_ff = 4 * d_model"""
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        d_ff = 4 * d_model
        self.w1 = _LinearW(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = _LinearW(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.w1(x)
        return self.w2(h * torch.sigmoid(h))  # SiLU(h) = h * sigmoid(h)
