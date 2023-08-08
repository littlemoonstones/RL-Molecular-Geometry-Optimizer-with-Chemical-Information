import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SelfAttentionLayer(nn.Module):

    def __init__(self, embeddin_size=48, num_heads=3, bias=True):
        super(SelfAttentionLayer, self).__init__()
        self.multi_attention = nn.MultiheadAttention(embeddin_size, num_heads, batch_first=True, bias=bias)

    def forward(self, x, mask):
        '''
        Inputs:
                -x : Input tensor of size [N_batch, N_atoms, D]
                -mask: Input tensor of size [N_batch, N_atoms]
        '''
        q = k = v = x
        out, _ = self.multi_attention(q, k, v, key_padding_mask=mask)
        
        return out

class AttentionLayer(nn.Module):

    def __init__(self, embeddin_size=48, num_heads=3, bias=True):
        super(AttentionLayer, self).__init__()
        assert embeddin_size%num_heads == 0

        # output_size = embeddin_size // num_heads
        self.query = nn.Linear(embeddin_size, embeddin_size, bias=bias)
        self.key = nn.Linear(embeddin_size, embeddin_size, bias=bias)
        self.value = nn.Linear(embeddin_size, embeddin_size, bias=bias)

        self.multi_attention = nn.MultiheadAttention(embeddin_size, num_heads, batch_first=True, bias=bias)

    def forward(self, x, mask):
        '''
        Inputs:
                -x : Input tensor of size [N_batch, N_atoms, D]
                -mask: Input tensor of size [N_batch, N_atoms]
        '''
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        out, _ = self.multi_attention(q, k, v, key_padding_mask=mask)
        
        return out


class ResidualNormLayer(nn.Module):

    def __init__(self, size):
        super(ResidualNormLayer, self).__init__()
        self.norm = nn.LayerNorm(size)

    def forward(self, x):
        out = self.norm(x) + x
        return out