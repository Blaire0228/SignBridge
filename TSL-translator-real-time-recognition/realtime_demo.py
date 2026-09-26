# -*- coding: utf-8 -*-
"""即時 webcam 手語辨識 demo。

流程：攝影機影格 → holistic 抽 55 點（與 extract_features.py 相同）
      → sliding window → 模型 → 手語詞。

需要舊版 mediapipe（holistic 在 0.10.15 後被移除），建議獨立 venv：
    pip install "mediapipe==0.10.14" opencv-python

執行：
    python realtime_demo.py --ckpt best_model.pt
    python realtime_demo.py --debug      # 顯示左右手偵測狀態，確認沒有左右顛倒
    按 q 離開。

★ 抽點一律用**未鏡像**的原始影格（跟訓練影片一致）；
  畫面上的鏡像只是為了顯示直覺，不會影響送進模型的資料。
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import config as C
from data import _resample_time as resample_time
from data import apply_feature_flags, build_features, normalize_sequence
from model import build_from_checkpoint


# ====================================================================== 推論
def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # 用訓練當時的特徵設定，避免 config 改過之後前後不一致
    apply_feature_flags(ckpt)
    model = build_from_checkpoint(ckpt).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"載入 {ckpt['arch']}（val acc={ckpt.get('val_acc', float('nan')):.3f}）")
    return model, ckpt["classes"], ckpt.get("seq_len", C.SEQ_LEN)


@torch.no_grad()
def predict_window(model, window: np.ndarray, device, seq_len: int = C.SEQ_LEN):
    """window: (T, 55, 3) 任意長度的原始關鍵點（含 NaN）。

    前處理順序與訓練時一字不差：正規化 → 重採樣到 seq_len → 組特徵。
    window 有多少格不重要，重要的是它涵蓋「一個完整手語詞」的時間長度。
    """
    coords, mask = normalize_sequence(window)
    coords = resample_time(coords, seq_len)
    mask = resample_time(mask[:, :, None], seq_len)[:, :, 0]
    x = build_features(coords, mask)[None, ...]
    logits = model(torch.from_numpy(x).float().to(device))
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(prob.argmax())
    return idx, float(prob[idx])


def pick_device(prefer="auto"):
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.OUT_DIR / C.ARCH / "best_model.pt"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threshold", type=float, default=C.PRED_THRESHOLD)
    ap.add_argument("--window-seconds", type=float, default=C.PRED_WINDOW_SECONDS,
                    help="sliding window 要涵蓋幾秒，需與訓練影片單詞長度相當")
    ap.add_argument("--window-frames", type=int, default=0,
                    help="直接指定 window 影格數；0 代表依實測 fps 自動換算")
    ap.add_argument("--no-mirror", action="store_true", help="關掉畫面鏡像顯示")
    ap.add_argument("--debug", action="store_true", help="顯示左右手偵測狀態")
    args = ap.parse_args()

    import cv2
    from extract import HolisticExtractor

    device = pick_device(args.device)
    model, classes, seq_len = load_model(Path(args.ckpt), device)
    extractor = HolisticExtractor()
    print(f"device={device}  類別數={len(classes)}  按 q 離開")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("打不開攝影機，檢查 macOS 的相機權限")

    # 先跑一段暖身量實際 fps，再決定 window 要存幾格
    WARMUP = 30
    window_frames = args.window_frames
    buf = deque(maxlen=window_frames or 10_000)
    recent = deque(maxlen=C.PRED_SMOOTH)
    frame_i, label, conf = 0, "", 0.0
    t_last, fps = time.time(), 0.0
    t_warm = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 抽點用原始影格（未鏡像），跟訓練影片一致
            pts = extractor(frame)
            buf.append(pts)
            frame_i += 1

            # 暖身結束後依實測 fps 換算 window 長度
            if window_frames == 0 and frame_i == WARMUP:
                measured = WARMUP / max(time.time() - t_warm, 1e-6)
                window_frames = max(seq_len, int(round(measured * args.window_seconds)))
                buf = deque(list(buf)[-window_frames:], maxlen=window_frames)
                print(f"實測 {measured:.1f} fps → window = {window_frames} 格 "
                      f"（約 {args.window_seconds} 秒），重採樣到 {seq_len} 格送進模型")

            if window_frames and len(buf) == window_frames and frame_i % C.PRED_STRIDE == 0:
                idx, p = predict_window(model, np.stack(buf), device, seq_len)
                recent.append(idx if p >= args.threshold else -1)
                if len(recent) == recent.maxlen and len(set(recent)) == 1 and recent[0] >= 0:
                    label, conf = classes[recent[0]], p

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now

            # 顯示才鏡像
            view = frame if args.no_mirror else cv2.flip(frame, 1)
            has_l = not np.isnan(pts[C.LHAND_SLICE]).all()
            has_r = not np.isnan(pts[C.RHAND_SLICE]).all()

            bar_h = 110 if args.debug else 70
            cv2.rectangle(view, (0, 0), (view.shape[1], bar_h), (0, 0, 0), -1)
            hud = f"{label}  {conf:.2f}" if label else "..."
            cv2.putText(view, hud, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 128), 2)
            cv2.putText(view, f"{fps:4.1f} fps  buf {len(buf)}/{window_frames or '?'}",
                        (view.shape[1] - 260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if args.debug:
                # 畫在黑底橫條上，綠＝偵測到、灰＝沒偵測到
                cv2.putText(view, "LEFT", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0) if has_l else (110, 110, 110), 2)
                cv2.putText(view, "RIGHT", (140, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0) if has_r else (110, 110, 110), 2)
                # 每秒也印一次到終端機，確保看得到
                if frame_i % max(1, int(round(max(fps, 1)))) == 0:
                    print(f"  LEFT={'OK' if has_l else '--'}  RIGHT={'OK' if has_r else '--'}"
                          f"   {label or '...'} {conf:.2f}", flush=True)

            cv2.imshow("TSL realtime", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()
