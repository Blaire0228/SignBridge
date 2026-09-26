# -*- coding: utf-8 -*-
"""即時抽點 —— 與組員的 extract_features.py 完全一致。

★ 這個檔案存在的唯一理由：訓練資料怎麼抽的，即時推論就要一模一樣抽。
  節點順序、左右手判定、缺值填 NaN 的方式，任何一點不同都會讓模型失準。
  所以這裡直接沿用 mp.solutions.holistic，而不是新版 Tasks API
  （Tasks API 的 HandLandmarker 用 handedness 分類器判左右，
    holistic 則是從 pose 推 ROI —— 兩者對左右手的認定會不一樣）。

因此本檔需要**舊版 mediapipe**（holistic 在 0.10.15 之後被移除）：
    pip install "mediapipe==0.10.14"
建議跟你的 TSL 生成專案分開用不同的 venv，避免和 0.10.35 打架。
請跟組員確認他跑 extract_features.py 時用的是哪個版本，直接對齊那個。
"""
from __future__ import annotations

import numpy as np

import config as C

# 與 extract_features.py 相同：pose 0:12, 左手 12:33, 右手 33:54, 虛擬頸點 54
POSE_INDICES = C.POSE_MP_INDICES
NUM_NODES = C.NUM_POINTS


def _missing(count):
    return [[np.nan, np.nan, np.nan] for _ in range(count)]


def extract_landmarks(results):
    """把 holistic 的結果轉成一格 (55, 3)。邏輯與 extract_features.py 相同。"""
    frame_data = []

    if results.pose_landmarks:
        for index in POSE_INDICES:
            lm = results.pose_landmarks.landmark[index]
            frame_data.append([lm.x, lm.y, lm.z])
    else:
        frame_data.extend(_missing(len(POSE_INDICES)))

    for landmarks in (results.left_hand_landmarks, results.right_hand_landmarks):
        if landmarks:
            frame_data.extend([lm.x, lm.y, lm.z] for lm in landmarks.landmark)
        else:
            frame_data.extend(_missing(21))

    if results.pose_landmarks:
        left = results.pose_landmarks.landmark[11]
        right = results.pose_landmarks.landmark[12]
        frame_data.append([
            (left.x + right.x) / 2.0,
            (left.y + right.y) / 2.0,
            (left.z + right.z) / 2.0,
        ])
    else:
        frame_data.extend(_missing(1))

    return frame_data


class HolisticExtractor:
    """包一層 context manager，供即時推論逐格呼叫。"""

    def __init__(self, model_complexity: int = 1):
        import mediapipe as mp

        if not hasattr(mp.solutions, "holistic"):
            raise RuntimeError(
                "這個 mediapipe 版本沒有 solutions.holistic。\n"
                '請安裝舊版：pip install "mediapipe==0.10.14"'
            )
        self._mp = mp
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame_bgr) -> np.ndarray:
        """輸入**未鏡像**的 BGR 影格，回傳 (55, 3)，未偵測到為 NaN。"""
        import cv2

        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        return np.asarray(extract_landmarks(results), dtype=np.float32)

    def close(self):
        self.holistic.close()
