# -*- coding: utf-8 -*-
"""55 節點骨架圖定義（ST-GCN 用）。

節點順序與 extract_features.py 完全一致：
    0..11   pose，對應 MediaPipe 的 [11,12,13,14,15,16,17,18,19,20,21,22]
    12..32  左手 21 點
    33..53  右手 21 點
    54      虛擬頸點（雙肩中點）

pose 那 12 點的實際意義：
    slot 0=左肩(11)  1=右肩(12)  2=左肘(13)  3=右肘(14)  4=左腕(15)  5=右腕(16)
    slot 6=左小指(17) 7=右小指(18) 8=左食指(19) 9=右食指(20) 10=左拇指(21) 11=右拇指(22)
"""
from __future__ import annotations

import numpy as np

import config as C

# ---------------------------------------------------------------- 節點編號
L_SHOULDER, R_SHOULDER = 0, 1
L_ELBOW, R_ELBOW = 2, 3
L_WRIST, R_WRIST = 4, 5
L_PINKY, R_PINKY = 6, 7
L_INDEX, R_INDEX = 8, 9
L_THUMB, R_THUMB = 10, 11
NECK = C.CENTER_IDX                      # 54
LH0 = C.LHAND_SLICE.start                # 12，左手腕
RH0 = C.RHAND_SLICE.start                # 33，右手腕

# ---------------------------------------------------------------- 邊
# 上半身骨架
POSE_EDGES = [
    (NECK, L_SHOULDER), (NECK, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_WRIST, L_PINKY), (L_WRIST, L_INDEX), (L_WRIST, L_THUMB),
    (R_WRIST, R_PINKY), (R_WRIST, R_INDEX), (R_WRIST, R_THUMB),
]

# MediaPipe 單手 21 點的標準連接
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),            # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),       # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),     # 無名指
    (13, 17), (17, 18), (18, 19), (19, 20),    # 小指
    (0, 17),                                    # 掌根
]


def build_edges():
    edges = list(POSE_EDGES)
    for off in (LH0, RH0):
        edges += [(a + off, b + off) for a, b in HAND_EDGES]
    # 把 pose 的手腕接到手部骨架的手腕，讓手臂和手掌是連通的
    edges += [(L_WRIST, LH0), (R_WRIST, RH0)]
    return edges


# ---------------------------------------------------------------- 鄰接矩陣
def _hop_distance(num_nodes: int, edges, max_hop: int = 1) -> np.ndarray:
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    hop = np.full((num_nodes, num_nodes), np.inf)
    powers = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive = (np.stack(powers) > 0)
    for d in range(max_hop, -1, -1):
        hop[arrive[d]] = d
    return hop


def _normalize(A: np.ndarray) -> np.ndarray:
    """D^-1 A，避免度數高的節點主導。"""
    deg = A.sum(axis=0)
    Dinv = np.zeros_like(A)
    nz = deg > 0
    Dinv[nz, nz] = deg[nz] ** -1
    return A @ Dinv


def build_adjacency(num_nodes: int = C.NUM_POINTS, strategy: str = "spatial") -> np.ndarray:
    """回傳 (K, V, V) 的鄰接矩陣堆疊。

    spatial（ST-GCN 原論文的分割方式）分成三組：
      0 根節點：自己 + 與自己離頸點等距的鄰居
      1 向心：比自己更靠近頸點的鄰居（代表軀幹方向的運動）
      2 離心：比自己更遠離頸點的鄰居（代表末端手指的運動）
    對手語很合理——同一個手勢，重點常在末端相對於軀幹怎麼動。
    """
    edges = build_edges()
    hop = _hop_distance(num_nodes, edges, max_hop=1)
    adjacency = (hop <= 1).astype(float)          # 含自環
    norm = _normalize(adjacency)

    dist_to_center = _hop_distance(num_nodes, edges, max_hop=num_nodes)[NECK]

    if strategy == "uniform":
        return norm[None, ...]
    if strategy != "spatial":
        raise ValueError(f"未知的 strategy: {strategy}")

    root = np.zeros_like(norm)
    close = np.zeros_like(norm)
    far = np.zeros_like(norm)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if hop[j, i] != 1 and i != j:
                continue
            di, dj = dist_to_center[i], dist_to_center[j]
            if dj == di:
                root[j, i] = norm[j, i]
            elif dj > di:
                far[j, i] = norm[j, i]
            else:
                close[j, i] = norm[j, i]
    return np.stack([root, close, far])


def build_parents(num_nodes: int = C.NUM_POINTS):
    """以頸點為根做 BFS，回傳 (parents, bfs_order)。

    parents[v] = v 的父節點索引，根節點為 -1。
    掌根那圈迴路在 BFS 時自然被展開成樹，不影響。
    骨向量與肢段縮放都需要這個樹狀結構。
    """
    adj = [[] for _ in range(num_nodes)]
    for i, j in build_edges():
        adj[i].append(j)
        adj[j].append(i)

    parents = np.full(num_nodes, -1, dtype=int)
    visited = np.zeros(num_nodes, dtype=bool)
    order, queue = [], [NECK]
    visited[NECK] = True
    while queue:
        v = queue.pop(0)
        order.append(v)
        for u in adj[v]:
            if not visited[u]:
                visited[u] = True
                parents[u] = v
                queue.append(u)
    return parents, order


def sanity_check():
    """檢查圖是連通的、沒有孤立節點。"""
    edges = build_edges()
    deg = np.zeros(C.NUM_POINTS, dtype=int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    isolated = np.where(deg == 0)[0]
    hop = _hop_distance(C.NUM_POINTS, edges, max_hop=C.NUM_POINTS)
    unreachable = np.where(~np.isfinite(hop[NECK]))[0]
    return {
        "num_edges": len(edges),
        "isolated_nodes": isolated.tolist(),
        "unreachable_from_neck": unreachable.tolist(),
        "max_hop_from_neck": float(np.nanmax(hop[NECK][np.isfinite(hop[NECK])])),
    }


if __name__ == "__main__":
    print(sanity_check())
    A = build_adjacency()
    print("鄰接矩陣 shape:", A.shape, " 每組非零數:", [int((a > 0).sum()) for a in A])
