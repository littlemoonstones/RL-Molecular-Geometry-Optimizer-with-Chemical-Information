import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SelfAttention(nn.Module):

    def __init__(self, input_size, hidden_size=16):
        '''
        This implementation assumes all the molecules of the same size i.e. the number of atoms
        '''
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.inp2Query = nn.Linear(input_size, hidden_size, bias=False)
        self.inp2Key = nn.Linear(input_size, hidden_size, bias=False)
        self.inp2Value = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x, mask):
        '''
        Inputs:
                -x : Input tensor of size [N_batch, N_atoms, D]
                -mask: Input tensor of size [N_batch, N_atoms]
        '''
        query = self.inp2Query(x)
        key = self.inp2Key(x)
        value = self.inp2Value(x)
        d_k = query.size(-1)
        mask = mask.unsqueeze(-1)
        mask = mask.transpose(-1,-2)
        attn_logits = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k)
        attn_logits = mask*attn_logits + (1-mask)*(-1e30)
        attn_dist = F.softmax(attn_logits, -1)
        out = torch.matmul(attn_dist, value)
        # print(out)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, input_size=48, num_heads=3):
        super(MultiHeadAttention, self).__init__()
        self.attnHeads = nn.ModuleList([SelfAttention(input_size, input_size//num_heads)
                                        for i in range(num_heads)])

    def forward(self, x, mask):
        '''
        Inputs:
                -x : Input tensor of size [N_batch, N_atoms, D]
                -mask: Input tensor of size [N_batch, N_atoms]
        '''
        num_heads = len(self.attnHeads)
        attn_outs = [self.attnHeads[i](x, mask) for i in range(num_heads)]
        out = torch.cat(attn_outs, -1)
        return out


class ResLayerNorm(nn.Module):

    def __init__(self, size):
        super(ResLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(size)

    def forward(self, x, op, mask=None):
        if mask is None:
            out = x + op(self.norm(x))
        else:
            out = x + op(self.norm(x), mask)

        return out