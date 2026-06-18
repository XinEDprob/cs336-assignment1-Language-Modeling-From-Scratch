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