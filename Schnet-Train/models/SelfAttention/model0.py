import torch
import torch.nn as nn
import torch.nn.functional as F
from models.components.self_attn import MultiHeadAttention, ResLayerNorm
import ptan
import numpy as np

class CustomModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 48, n_head = 3):
        super(CustomModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.mh_self_attn1 = MultiHeadAttention(hidden_size, n_head)
        self.mh_addnorm1 = ResLayerNorm(hidden_size) 
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # bool_mask = (1-mask).bool()
        out = F.relu(self.input_layer(x))
        out = self.mh_addnorm1(out, self.mh_self_attn1, mask)
        out = self.output_layer(out)

        return out 