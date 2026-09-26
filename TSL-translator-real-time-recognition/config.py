# -*- coding: utf-8 -*-
"""全域設定：訓練與即時推論共用，改這裡就好。"""
from pathlib import Path

# ---------------------------------------------------------------- 路徑
PROJECT_DIR = Path(__file__).resolve().parent
# 把 Google Drive 的 NPY_Dataset 下載到本機後，放在這個路徑
DATA_ROOT = PROJECT_DIR / "data" / "NPY_Dataset"
OUT_DIR = PROJECT_DIR / "runs"

# ---------------------------------------------------------------- 關鍵點佈局
# 每個 .npy 形狀為 (T, 55, 3)。55 = 12 上半身 pose + 21 左手 + 21 右手 + 1 虛擬頸點。
# ✓ 已與組員的 extract_features.py 核對過，完全一致。
NUM_POINTS = 55
POSE_SLICE = slice(0, 12)      # 上半身 pose
LHAND_SLICE = slice(12, 33)    # 左手 21 點
RHAND_SLICE = slice(33, 54)    # 右手 21 點
CENTER_IDX = 54                # 虛擬頸點＝雙肩中點（正規化原點）
LSHOULDER_IDX = 0
RSHOULDER_IDX = 1

# 從 MediaPipe Pose 的 33 點裡挑的 12 點（＝ extract_features.py 的 POSE_INDICES）。
POSE_MP_INDICES = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# ---------------------------------------------------------------- 前處理
# 實際資料：影格數 min=55 / median=150 / max=179，所以重採樣到 64 格
# （約 2.3 倍下採樣，保留足夠的手部動作細節，又不會讓訓練太慢）
SEQ_LEN = 64
MIN_FRAMES = 8        # 少於這個影格數的樣本會先補到這個長度
USE_XY_ONLY = True    # MediaPipe 的 z 噪音大，預設只用 x,y（設 False 則保留全部維度）
ADD_VELOCITY = True   # 加入相鄰影格差分（動作速度），對手語很有幫助
ADD_MASK = True       # 加入「這一點這一格有沒有被偵測到」的 0/1 通道

# --- 跨人泛化用的特徵（骨向量與關節角度對體型差異較不敏感）
ADD_BONE = True       # 每個節點指向父節點的向量（長度隱含肢段比例）
ADD_ANGLE = True      # 該節點在骨架樹上與父、祖父形成的夾角（純角度，與體型無關）

# ---------------------------------------------------------------- 訓練
SEED = 42
VAL_RATIO = 0.2
BATCH_SIZE = 32
EPOCHS = 120
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 25

# ---------------------------------------------------------------- 模型
ARCH = "stgcn"
DROPOUT = 0.3

# ST-GCN
GRAPH_STRATEGY = "spatial"      # "spatial"（三組分割）或 "uniform"
STGCN_CHANNELS = (64, 64, 128, 128)
STGCN_STRIDES = (1, 1, 2, 1)    # 2 代表時間軸下採樣一半
STGCN_EDGE_IMPORTANCE = True    # 讓模型自己學每條骨架邊的重要性

# ---------------------------------------------------------------- 資料增強
# 目標不只是防過擬合，而是 domain randomization ——
# 用單一拍攝者的資料合成出「各種體型、各種取景、各種偵測品質」的假樣本，
# 讓模型不要依賴那些會隨人改變的線索。
AUG_SCALE = 0.20       # 隨機縮放 ±20%
AUG_ROTATE_DEG = 18.0  # 隨機旋轉 ±18 度
AUG_SHIFT = 0.10       # 隨機平移
AUG_NOISE = 0.015      # 高斯雜訊
AUG_TIME_WARP = 0.3    # 隨機加減速 ±30%
AUG_FRAME_DROP = 0.08  # 隨機丟影格比例
AUG_MIRROR = False     # 左右鏡像：手語中左右手分工有意義，預設關閉

# --- 模擬跨人/跨環境的差異
AUG_LIMB_SCALE = 0.15     # 各肢段長度獨立隨機縮放 ±15%，模擬不同體型
AUG_JOINT_DROP = 0.15     # 隨機關節在隨機時段「偵測失敗」，模擬 MediaPipe 掉點
AUG_HAND_DROP = 0.15      # 整隻手短暫消失的機率（實測即時環境常發生）
AUG_TEMPORAL_CROP = 0.25  # 隨機裁掉頭尾最多 25%，模擬滑動視窗沒對齊完整動作

# ---------------------------------------------------------------- 即時推論
# ★ 關鍵：訓練時每支影片不管多長都被時間正規化成 SEQ_LEN 格，
#   所以模型學到的是「一個完整手語詞從頭到尾」的樣子。
#   即時推論的 sliding window 必須涵蓋**同樣長度的時間**（而不是同樣的影格數），
#   否則模型會看到一個被快轉 3 倍的動作，準確率直接崩掉。
#
# 訓練影片中位數 150 格。若原始影片是 30 fps → 一個詞約 5.0 秒。
# ★ 請跟組員確認原始 mp4 的 fps，再改這個數字：PRED_WINDOW_SECONDS = 150 / fps
PRED_WINDOW_SECONDS = 5.0
PRED_STRIDE = 4            # 每幾格推論一次
PRED_THRESHOLD = 0.75      # 信心門檻，低於此不輸出
PRED_SMOOTH = 3            # 連續幾次相同預測才確認
