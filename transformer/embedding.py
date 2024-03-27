import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Embedding(nn.Embedding):
    """Lookup table for vocab embedding, size: d_vocab x d_model
        Output: Batch size x seq len x embedding dim
    """
    def __init__(self, d_model, d_vocab) -> None:
        super().__init__()
        self.embed = nn.Embedding(d_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # return x's embedding vector（times math.sqrt(d_model)）
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len) -> None:
        super().__init__(*args, **kwargs)