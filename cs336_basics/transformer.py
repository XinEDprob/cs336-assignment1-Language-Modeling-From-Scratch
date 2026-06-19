import torch


class LinearTransform(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = (2.0/(in_features + out_features))**0.5
        weight_data = torch.empty(out_features, in_features, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(weight_data, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        self.weight = torch.nn.parameter.Parameter(weight_data)

    def forward(self, x):
        return x @ self.weight.T
    

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        sigma = 1
        embeddings_data = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embeddings_data, mean=0, std=sigma, a=-3, b=3)
        self.embeddings = torch.nn.parameter.Parameter(embeddings_data)
    def forward(self, x):
        return self.embeddings[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.dtype = dtype
        g_empty = torch.ones(d_model, device=device, dtype=dtype)
        self.g = torch.nn.parameter.Parameter(g_empty)
    def forward(self, x):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        rms_x = (torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)**0.5
        x_norm = x / rms_x * self.g
        return x_norm.to(orig_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff=-1, device=None, dtype=None):
        super().__init__()
        if d_ff == -1:
            d_ff_raw = d_model * 8 / 3
            if d_ff_raw % 64 <= 21.5:
                d_ff = int(d_ff_raw // 64 * 64)
            else:
                d_ff = int((d_ff_raw // 64 + 1) * 64)
        self.linear1 = LinearTransform(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = LinearTransform(d_model, d_ff, device=device, dtype=dtype)
        self.linear3 = LinearTransform(d_ff, d_model, device=device, dtype=dtype)
    def forward(self, x):
        w1_out = self.linear1(x)
        w2_out = self.linear2(x)
        gate = w1_out * torch.sigmoid(w1_out)  # Swish activation
        return self.linear3(gate*w2_out)
    

class ROPE(torch.nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k), token_positions: (..., seq_len)
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]

        x1 = x[..., ::2]   # even indices
        x2 = x[..., 1::2]  # odd indices

        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)


def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    return softmax(scores, dim=-1) @ V


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta):
        super().__init__()
        d_k = d_v = d_model // num_heads
        assert d_k > 0, "d_k should be larger than 0"

        empty_q = torch.empty(d_model, d_model)
        empty_k = torch.empty(d_model, d_model)
        empty_v = torch.empty(d_model, d_model)


        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        init_std_k = (2/(self.d_k+d_model))**0.5
        init_std_v = (2/(d_v+d_model))**0.5

        self.weight_q = torch.nn.parameter.Parameter(torch.nn.init.trunc_normal_(empty_q, 0, init_std_k, a=-3*init_std_k, b=3*init_std_k))
        self.weight_k = torch.nn.parameter.Parameter(torch.nn.init.trunc_normal_(empty_k, 0, init_std_k, a=-3*init_std_k, b=3*init_std_k))
        self.weight_v = torch.nn.parameter.Parameter(torch.nn.init.trunc_normal_(empty_v, 0, init_std_v, a=-3*init_std_v, b=3*init_std_v))
        self.output_proj = LinearTransform(d_model, d_model)

        self.rope = ROPE(theta=theta, d_k=self.d_k, max_seq_len=self.max_seq_len)

    def forward(self, x, token_positions=None):
        batch_size, seq_len, _ = x.shape
        Q = x @ self.weight_q.T
        K = x @ self.weight_k.T
        V = x @ self.weight_v.T

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if token_positions is not None:
            Q = self.rope(Q, token_positions.unsqueeze(1))
            K = self.rope(K, token_positions.unsqueeze(1))

        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device)
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        attn_output = self.output_proj(attn_output)  # Output projection using the same weights as V projection
        return attn_output


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x, token_positions=None):
        batch_size, seq_len, _ = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x 

    
class Transformer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, context_length, theta, vocab_size, num_layers):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = LinearTransform(d_model, vocab_size)

    def forward(self, x):
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)
    