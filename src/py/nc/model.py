"""
nc/model.py
===========
The trained model architecture, copied VERBATIM from TCG-multilead-KL.py so that
training and inference instantiate the identical network. The EWS loads Run A's
checkpoint into TemporalTCG(input_channels=8, base_filters=32).

To change the architecture in the training script, change it here too
(or, better, have the training script import from this module to keep one copy).

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_NUM_CHANNELS = 8


class _ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell — orthogonal init + forget-gate bias=1 + cell clamp."""
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim,
            kernel_size=kernel_size, padding=pad, bias=True,
        )
        nn.init.orthogonal_(self.conv.weight, gain=0.5)
        nn.init.zeros_(self.conv.bias)
        self.conv.bias.data[hidden_dim:2 * hidden_dim].fill_(1.0)  # forget gate

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = torch.clamp(f * c + i * g, -10.0, 10.0)
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class TemporalTCG(nn.Module):
    """
    Convolutional LSTM encoder that jointly processes all T timesteps and
    outputs a spatial genesis logit map from the final hidden state.

    Input:  (B, T, C, H, W)  — T atmospheric snapshots ending at genesis-lead
    Output: (B, 1, H, W)     — raw logits (no activation)
    """

    def __init__(self, input_channels: int = DEFAULT_NUM_CHANNELS,
                 base_filters: int = 64):
        super().__init__()
        bf = base_filters

        self.embed = nn.Sequential(
            nn.Conv2d(input_channels, bf, kernel_size=1, bias=False),
            nn.BatchNorm2d(bf),
            nn.ReLU(inplace=True),
        )

        self.clstm1 = _ConvLSTMCell(bf,      bf,      kernel_size=3)
        self.clstm2 = _ConvLSTMCell(bf,      bf * 2,  kernel_size=3)

        self.t_attn = nn.Sequential(
            nn.Conv2d(bf * 2, 1, kernel_size=1, bias=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(bf * 2, bf * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bf * 2, bf,     kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bf),
            nn.ReLU(inplace=True),
            nn.Conv2d(bf,     1,      kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, C, H, W)  ->  (B, 1, H, W)"""
        B, T, C, H, W = x.shape
        device = x.device

        emb = [self.embed(x[:, t]) for t in range(T)]

        h1 = torch.zeros(B, self.clstm1.hidden_dim, H, W, device=device)
        c1 = torch.zeros(B, self.clstm1.hidden_dim, H, W, device=device)
        out1 = []
        for t in range(T):
            h1, c1 = self.clstm1(emb[t], h1, c1)
            out1.append(h1)

        h2 = torch.zeros(B, self.clstm2.hidden_dim, H, W, device=device)
        c2 = torch.zeros(B, self.clstm2.hidden_dim, H, W, device=device)
        out2 = []
        for t in range(T):
            h2, c2 = self.clstm2(out1[t], h2, c2)
            out2.append(h2)

        seq = torch.stack(out2, dim=1)             # (B, T, bf*2, H, W)
        scores = [self.t_attn(seq[:, t]) for t in range(T)]
        scores = torch.stack(scores, dim=1)        # (B, T, 1, H, W)
        weights = torch.softmax(scores, dim=1)     # softmax over T
        attended = (seq * weights).sum(dim=1)      # (B, bf*2, H, W)

        return self.refine(attended)               # (B, 1, H, W)
