# -*- coding: utf-8 -*-
"""資料載入、前處理、增強。

原始資料：DATA_ROOT/<編號>.<英文詞>/<classID><sampleID>.npy，每個檔案 shape = (T, 55, 3)。
前處理鏈：NaN 補值 → 以頸點置中、肩寬正規化 → 時間軸重採樣 → 速度/遮罩特徵。
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

import config as C


# ====================================================================== 掃描資料集
def scan_dataset(root: Path):
    """回傳 (檔案路徑清單, 標籤索引清單, 類別名稱清單)。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"找不到資料夾 {root}\n"
            "請先把雲端的 NPY_Dataset 下載到本機，或修改 config.py 的 DATA_ROOT。"
        )

    def sort_key(name: str):
        m = re.match(r"(\d+)", name)
        return (int(m.group(1)) if m else 10**9, name)

    class_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()], key=lambda d: sort_key(d.name)
    )
    if not class_dirs:
        raise RuntimeError(f"{root} 底下沒有任何類別資料夾")

    files, labels, classes = [], [], []
    for d in class_dirs:
        paths = sorted(glob.glob(str(d / "*.npy")))
        if not paths:
            print(f"[警告] 類別 {d.name} 沒有 .npy，略過")
            continue
        classes.append(d.name)
        for p in paths:
            files.append(p)
            labels.append(len(classes) - 1)
    return files, np.array(labels, dtype=np.int64), classes


# ====================================================================== 前處理
def _interp_nan_along_time(seq: np.ndarray) -> np.ndarray:
    """對每個關鍵點沿時間軸線性內插補 NaN；整段皆 NaN 的點補 0。"""
    T, P, D = seq.shape
    flat = seq.reshape(T, P * D).copy()
    t = np.arange(T)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        good = ~np.isnan(col)
        if good.all():
            continue
        if good.sum() == 0:
            col[:] = 0.0  # 這個點整段沒偵測到（例如全程沒出現的那隻手）
        else:
            col[~good] = np.interp(t[~good], t[good], col[good])
        flat[:, j] = col
    return flat.reshape(T, P, D)


def _resample_time(seq: np.ndarray, out_len: int) -> np.ndarray:
    """把 (T, P, D) 沿時間軸線性重採樣成 (out_len, P, D)。"""
    T = seq.shape[0]
    if T == out_len:
        return seq
    src = np.linspace(0.0, T - 1, num=T)
    dst = np.linspace(0.0, T - 1, num=out_len)
    flat = seq.reshape(T, -1)
    out = np.empty((out_len, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst, src, flat[:, j])
    return out.reshape(out_len, seq.shape[1], seq.shape[2])


def normalize_sequence(raw: np.ndarray):
    """把單一原始序列標準化成與拍攝距離、身體位置無關的座標。

    Args:
        raw: (T, 55, C) 原始關鍵點，可能含 NaN。
    Returns:
        coords: (T, 55, D) 已置中/縮放/補值的座標
        mask:   (T, 55)    1 = 該格該點原本有偵測到
    """
    seq = np.asarray(raw, dtype=np.float32)
    if seq.ndim != 3:
        raise ValueError(f"預期 (T, P, C) 三維陣列，收到 {seq.shape}")
    if C.USE_XY_ONLY and seq.shape[2] >= 2:
        seq = seq[:, :, :2]

    mask = (~np.isnan(seq).any(axis=2)).astype(np.float32)  # (T, P)

    # --- 原點：頸點（缺失時退回所有有效點的平均）
    if seq.shape[1] > C.CENTER_IDX:
        center = seq[:, C.CENTER_IDX, :]                     # (T, D)
    else:
        center = np.nanmean(seq, axis=1)
    bad = np.isnan(center).any(axis=1)
    if bad.any():
        fallback = np.nanmean(seq, axis=1)
        center = np.where(bad[:, None], fallback, center)
    center = np.nan_to_num(center, nan=0.0)

    # --- 尺度：肩寬中位數（缺失時退回關鍵點座標的標準差）
    if seq.shape[1] > max(C.LSHOULDER_IDX, C.RSHOULDER_IDX):
        span = np.linalg.norm(
            seq[:, C.LSHOULDER_IDX, :] - seq[:, C.RSHOULDER_IDX, :], axis=-1
        )
        scale = np.nanmedian(span)
    else:
        scale = np.nan
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.nanstd(seq)) or 1.0

    seq = (seq - center[:, None, :]) / float(scale)
    seq = _interp_nan_along_time(seq)
    return seq.astype(np.float32), mask


_PARENTS, _BFS_ORDER = None, None


def _parents():
    """延遲載入骨架樹（避免 data.py 與 graph.py 的循環匯入）。"""
    global _PARENTS, _BFS_ORDER
    if _PARENTS is None:
        from graph import build_parents
        _PARENTS, _BFS_ORDER = build_parents(C.NUM_POINTS)
    return _PARENTS, _BFS_ORDER


def build_features(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(T, P, D) + (T, P) → (T, P, Ch) 每個節點自己的特徵向量。

    保留「節點」維度，讓 ST-GCN 能直接使用每個骨架節點的特徵。

    通道組成（依 config 開關）：
      座標 x,y          位置資訊，但仍帶有個人習慣
      速度 vx,vy        動作方向與快慢
      遮罩 mask         這一點這一格是不是補出來的
      骨向量 bx,by      指向父節點，與絕對位置無關
      關節角度 angle    純角度，與體型、肢長完全無關 ← 跨人泛化的關鍵
    """
    T, P, D = coords.shape
    parts = [coords]
    if C.ADD_VELOCITY:
        vel = np.zeros_like(coords)
        vel[1:] = coords[1:] - coords[:-1]
        parts.append(vel)
    if C.ADD_MASK:
        parts.append(mask[:, :, None])

    if C.ADD_BONE or C.ADD_ANGLE:
        parents, _ = _parents()
        par = np.where(parents >= 0, parents, np.arange(P))
        bone = coords - coords[:, par, :]          # (T, P, D)，根節點為 0
        if C.ADD_BONE:
            parts.append(bone)
        if C.ADD_ANGLE:
            # 該節點的骨向量 與 其父節點的骨向量 之間的夾角餘弦
            parent_bone = bone[:, par, :]
            n1 = np.linalg.norm(bone, axis=2, keepdims=True)
            n2 = np.linalg.norm(parent_bone, axis=2, keepdims=True)
            cos = (bone * parent_bone).sum(axis=2, keepdims=True) / np.maximum(n1 * n2, 1e-6)
            parts.append(np.clip(cos, -1.0, 1.0))

    return np.concatenate(parts, axis=2).astype(np.float32)


def node_channels() -> int:
    """每個節點每一格有幾個通道。"""
    d = 2 if C.USE_XY_ONLY else 3
    ch = d
    if C.ADD_VELOCITY:
        ch += d
    if C.ADD_MASK:
        ch += 1
    if C.ADD_BONE:
        ch += d
    if C.ADD_ANGLE:
        ch += 1
    return ch


def feature_flags() -> dict:
    """存進 checkpoint，推論時據此還原相同的特徵設定。"""
    return dict(USE_XY_ONLY=C.USE_XY_ONLY, ADD_VELOCITY=C.ADD_VELOCITY,
                ADD_MASK=C.ADD_MASK, ADD_BONE=C.ADD_BONE, ADD_ANGLE=C.ADD_ANGLE,
                SEQ_LEN=C.SEQ_LEN)


def flags_from_ckpt(ckpt: dict) -> dict:
    """從 checkpoint 取出特徵設定，並相容於還沒有骨向量/角度通道的舊檔。"""
    if ckpt.get("feature_flags"):
        return ckpt["feature_flags"]
    return dict(
        USE_XY_ONLY=ckpt.get("use_xy_only", True),
        ADD_VELOCITY=ckpt.get("add_velocity", True),
        ADD_MASK=ckpt.get("add_mask", True),
        ADD_BONE=False, ADD_ANGLE=False,     # 舊版沒有這兩組通道
        SEQ_LEN=ckpt.get("seq_len", C.SEQ_LEN),
    )


def apply_feature_flags(ckpt_or_flags):
    """把特徵設定切換回該 checkpoint 訓練當時的版本。

    傳入完整 checkpoint 或 flags dict 皆可。這一步很重要 ——
    config 改過之後，舊權重的通道數會對不上。
    """
    flags = ckpt_or_flags
    if isinstance(ckpt_or_flags, dict) and "state_dict" in ckpt_or_flags:
        flags = flags_from_ckpt(ckpt_or_flags)
    for k, v in (flags or {}).items():
        if hasattr(C, k):
            setattr(C, k, v)


# ====================================================================== 資料增強
def limb_scale(coords: np.ndarray, rng: np.random.Generator, amount: float) -> np.ndarray:
    """沿骨架樹隨機縮放每一段肢段的長度，模擬不同體型的人。

    做法：由根節點往外走，把每個節點相對父節點的骨向量乘上隨機倍率，
    再依新的骨向量重建位置（等同前向運動學）。動作形狀保留，身體比例改變。
    """
    parents, order = _parents()
    P = coords.shape[1]
    factor = 1.0 + rng.uniform(-amount, amount, size=P)
    bone = coords - coords[:, np.where(parents >= 0, parents, np.arange(P)), :]
    out = coords.copy()
    for v in order:
        p = parents[v]
        if p < 0:
            continue
        out[:, v] = out[:, p] + bone[:, v] * factor[v]
    return out


def _drop_span(coords, mask, rng, node_idx):
    """讓指定節點在一段隨機時間內「偵測失敗」。

    模擬真實情形：mask 標 0，座標用該時段前後的值線性內插填補
    —— 這正是 normalize_sequence 遇到 NaN 時的處理方式。
    """
    T = coords.shape[0]
    span = max(2, int(T * rng.uniform(0.1, 0.5)))
    start = int(rng.integers(0, max(1, T - span)))
    end = min(T, start + span)
    if start == 0 and end >= T:
        coords[:, node_idx] = 0.0
    else:
        seg = coords[start:end, node_idx]
        a = coords[max(start - 1, 0), node_idx]
        b = coords[min(end, T - 1), node_idx]
        # ramp 的維度要跟著 node_idx 是單一節點還是一段 slice 調整
        ramp = np.linspace(0, 1, end - start).reshape((-1,) + (1,) * (seg.ndim - 1))
        coords[start:end, node_idx] = a[None] * (1 - ramp) + b[None] * ramp
    mask[start:end, node_idx] = 0.0
    return coords, mask


def augment(coords: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """在正規化座標空間做幾何 + 時間增強。coords: (T, P, D)"""
    T, P, D = coords.shape

    # 體型：隨機改變肢段比例（domain randomization 的核心）
    if C.AUG_LIMB_SCALE > 0:
        coords = limb_scale(coords, rng, C.AUG_LIMB_SCALE)

    # 時間：隨機裁掉頭尾，模擬即時視窗沒對齊完整動作
    if C.AUG_TEMPORAL_CROP > 0 and T > C.MIN_FRAMES * 2:
        keep_ratio = 1.0 - rng.uniform(0, C.AUG_TEMPORAL_CROP)
        new_T = max(C.MIN_FRAMES, int(T * keep_ratio))
        start = int(rng.integers(0, T - new_T + 1))
        coords, mask = coords[start:start + new_T], mask[start:start + new_T]
        T = new_T

    # 偵測失敗：整隻手短暫消失
    if C.AUG_HAND_DROP > 0 and rng.random() < C.AUG_HAND_DROP:
        sl = C.LHAND_SLICE if rng.random() < 0.5 else C.RHAND_SLICE
        coords, mask = coords.copy(), mask.copy()
        coords, mask = _drop_span(coords, mask, rng, slice(sl.start, sl.stop))

    # 偵測失敗：零星關節掉點
    if C.AUG_JOINT_DROP > 0:
        n_drop = int(P * C.AUG_JOINT_DROP * rng.random())
        if n_drop > 0:
            coords, mask = coords.copy(), mask.copy()
            for v in rng.choice(P, size=n_drop, replace=False):
                coords, mask = _drop_span(coords, mask, rng, int(v))

    # 時間：隨機加減速
    if C.AUG_TIME_WARP > 0:
        factor = 1.0 + rng.uniform(-C.AUG_TIME_WARP, C.AUG_TIME_WARP)
        new_T = max(C.MIN_FRAMES, int(round(T * factor)))
        coords = _resample_time(coords, new_T)
        mask = _resample_time(mask[:, :, None], new_T)[:, :, 0]
        T = new_T

    # 時間：隨機丟影格（模擬掉幀）
    if C.AUG_FRAME_DROP > 0 and T > C.MIN_FRAMES * 2:
        keep = rng.random(T) > C.AUG_FRAME_DROP
        if keep.sum() >= C.MIN_FRAMES:
            coords, mask = coords[keep], mask[keep]

    # 幾何：旋轉（只作用在 x,y 平面）
    if C.AUG_ROTATE_DEG > 0 and D >= 2:
        th = np.deg2rad(rng.uniform(-C.AUG_ROTATE_DEG, C.AUG_ROTATE_DEG))
        c, s = np.cos(th), np.sin(th)
        xy = coords[:, :, :2]
        coords = coords.copy()
        coords[:, :, 0] = xy[:, :, 0] * c - xy[:, :, 1] * s
        coords[:, :, 1] = xy[:, :, 0] * s + xy[:, :, 1] * c

    # 幾何：縮放 / 平移 / 雜訊
    if C.AUG_SCALE > 0:
        coords = coords * (1.0 + rng.uniform(-C.AUG_SCALE, C.AUG_SCALE))
    if C.AUG_SHIFT > 0:
        coords = coords + rng.uniform(-C.AUG_SHIFT, C.AUG_SHIFT, size=(1, 1, D))
    if C.AUG_NOISE > 0:
        coords = coords + rng.normal(0, C.AUG_NOISE, size=coords.shape)

    # 左右鏡像（預設關閉，見 config 說明）
    if C.AUG_MIRROR and rng.random() < 0.5 and P == C.NUM_POINTS:
        coords = coords.copy()
        coords[:, :, 0] *= -1
        lh, rh = C.LHAND_SLICE, C.RHAND_SLICE
        coords[:, lh], coords[:, rh] = coords[:, rh].copy(), coords[:, lh].copy()
        mask = mask.copy()
        mask[:, lh], mask[:, rh] = mask[:, rh].copy(), mask[:, lh].copy()

    return coords.astype(np.float32), mask.astype(np.float32)


# ====================================================================== Dataset
class SignDataset(Dataset):
    """把 .npy 全部預先載入並前處理好，放在記憶體裡。

    前處理（正規化、補值）跟增強無關，所以只做一次；
    訓練用的增強在 __getitem__ 才即時套用。
    用 subset() 取子集時會共用同一份記憶體，不會重複載入。
    """

    def __init__(self, files=None, labels=None, train: bool = False,
                 seed: int = C.SEED, _items=None, _labels=None, verbose=True):
        self.train = train
        self.rng = np.random.default_rng(seed)

        if _items is not None:                      # 由 subset() 建立的檢視
            self.items, self.labels = _items, _labels
            return

        self.labels = np.asarray(labels, dtype=np.int64)
        self.items = []
        n = len(files)
        for k, p in enumerate(files):
            raw = np.load(p)
            if raw.shape[0] < C.MIN_FRAMES:
                raw = _resample_time(np.asarray(raw, dtype=np.float32), C.MIN_FRAMES)
            self.items.append(normalize_sequence(raw))
            if verbose and n > 500 and (k + 1) % 500 == 0:
                print(f"  前處理 {k + 1}/{n} …", flush=True)

    def subset(self, indices, train: bool, seed: int = C.SEED) -> "SignDataset":
        """共用已前處理好的資料，只換索引與是否增強。"""
        idx = list(indices)
        return SignDataset(
            train=train, seed=seed,
            _items=[self.items[i] for i in idx],
            _labels=self.labels[idx],
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        coords, mask = self.items[i]
        if self.train:
            coords, mask = augment(coords, mask, self.rng)
        coords = _resample_time(coords, C.SEQ_LEN)
        mask = _resample_time(mask[:, :, None], C.SEQ_LEN)[:, :, 0]
        return build_features(coords, mask), self.labels[i]


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int):
    """每個類別各留 val_ratio 當驗證集，至少留 1 筆。"""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))
