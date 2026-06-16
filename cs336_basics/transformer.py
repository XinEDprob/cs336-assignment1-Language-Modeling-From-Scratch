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