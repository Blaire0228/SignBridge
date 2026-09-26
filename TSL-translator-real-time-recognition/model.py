# -*- coding: utf-8 -*-
"""ST-GCN model used by the word-level sign recognizer.

Input shape: (B, T, V, Ch), where V=55 contains upper-body and hand
landmarks. Spatial graph convolution follows the skeleton topology, while
temporal convolution learns how each sign evolves across frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from graph import build_adjacency


class SpatialGraphConv(nn.Module):
    """圖卷積：每組鄰接矩陣各學一套權重，再沿著骨架邊聚合。"""

    def __init__(self, in_ch, out_ch, num_subsets):
        super().__init__()
        self.k = num_subsets
        self.conv = nn.Conv2d(in_ch, out_ch * num_subsets, kernel_size=1)

    def forward(self, x, A):                       # x: (B, C, T, V), A: (K, V, V)
        x = self.conv(x)
        B, KC, T, V = x.shape
        x = x.view(B, self.k, KC // self.k, T, V)
        # 沿著邊把鄰居的特徵聚合過來
        x = torch.einsum("bkctv,kvw->bctw", x, A)
        return x.contiguous()


class STGCNBlock(nn.Module):
    """空間圖卷積 → 時間卷積 → 殘差。"""

    def __init__(self, in_ch, out_ch, num_subsets, stride=1, dropout=0.3, t_kernel=9):
        super().__init__()
        self.gcn = SpatialGraphConv(in_ch, out_ch, num_subsets)
        self.bn_g = nn.BatchNorm2d(out_ch)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (t_kernel, 1), (stride, 1), ((t_kernel - 1) // 2, 0)),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )
        if in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, (stride, 1)), nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        h = self.relu(self.bn_g(self.gcn(x, A)))
        return self.relu(self.tcn(h) + res)


class SignSTGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, num_classes,
                 channels=(64, 64, 128, 128), strides=(1, 1, 2, 1),
                 dropout=0.3, strategy="spatial", edge_importance=True):
        super().__init__()
        A = build_adjacency(num_nodes, strategy)
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        k = self.A.shape[0]

        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        blocks, prev = [], in_channels
        for ch, st in zip(channels, strides):
            blocks.append(STGCNBlock(prev, ch, k, stride=st, dropout=dropout))
            prev = ch
        self.blocks = nn.ModuleList(blocks)

        # 讓模型自己學「哪幾條骨架邊比較重要」，ST-GCN 原論文的做法
        if edge_importance:
            self.edge_weight = nn.ParameterList(
                [nn.Parameter(torch.ones_like(self.A)) for _ in blocks]
            )
        else:
            self.edge_weight = [1.0] * len(blocks)

        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(prev, num_classes))

    def forward(self, x):                          # x: (B, T, V, Ch)
        B, T, V, Ch = x.shape
        h = x.permute(0, 3, 1, 2).contiguous()     # (B, Ch, T, V)
        h = self.data_bn(h.permute(0, 1, 3, 2).reshape(B, Ch * V, T))
        h = h.view(B, Ch, V, T).permute(0, 1, 3, 2).contiguous()

        for block, w in zip(self.blocks, self.edge_weight):
            h = block(h, self.A * w)

        h = h.mean(dim=(2, 3))                     # 時間與節點都做平均池化
        return self.head(h)


def build_from_checkpoint(ckpt: dict):
    """給即時推論用：照 checkpoint 裡存的設定還原模型。"""
    import types

    if ckpt.get("arch") != "stgcn":
        raise ValueError("This release only supports ST-GCN checkpoints")
    cfg = types.SimpleNamespace(**ckpt["model_cfg"])
    return SignSTGCN(
        ckpt["num_nodes"],
        ckpt["in_channels"],
        len(ckpt["classes"]),
        channels=cfg.STGCN_CHANNELS,
        strides=cfg.STGCN_STRIDES,
        dropout=cfg.DROPOUT,
        strategy=cfg.GRAPH_STRATEGY,
        edge_importance=cfg.STGCN_EDGE_IMPORTANCE,
    )
