from __future__ import annotations

import json
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


APP_DIR = Path(__file__).resolve().parent.parent
MODEL_ASSET_DIR = APP_DIR / "TSL-translator-real-time-recognition"
CHECKPOINT_PATH = MODEL_ASSET_DIR / "best_model.pt"
SIGN_WORDS_PATH = APP_DIR / "sign_language_words.json"

# The model project uses top-level imports (for example ``import config``), so
# expose its directory while importing its inference modules.
model_asset_path = str(MODEL_ASSET_DIR)
if model_asset_path not in sys.path:
    sys.path.insert(0, model_asset_path)

from data import _resample_time as resample_time  # noqa: E402
from data import apply_feature_flags, build_features, normalize_sequence  # noqa: E402
from extract import HolisticExtractor  # noqa: E402
from model import build_from_checkpoint  # noqa: E402


MIN_INPUT_FRAMES = 8
MAX_LANDMARK_FRAMES = 64
DEFAULT_CONFIDENCE_THRESHOLD = 0.30


class SignRecognitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecognitionCandidate:
    label: str
    raw_label: str
    confidence: float
    confidence_percent: float


@dataclass(frozen=True)
class RecognitionResult:
    label: str | None
    raw_label: str
    confidence: float
    confidence_percent: float
    accepted: bool
    frame_count: int
    candidates: list[RecognitionCandidate]
    message: str | None = None
    pose_frame_count: int | None = None
    left_hand_frame_count: int | None = None
    right_hand_frame_count: int | None = None
    model_source: str = "TSL-translator-real-time-recognition/best_model.pt"

    def to_dict(self) -> dict:
        return asdict(self)


class StreamingRecognitionSession:
    """Accumulate MediaPipe landmarks while the phone camera is active."""

    def __init__(self, service: "SignRecognitionService") -> None:
        self._service = service
        self._extractor = HolisticExtractor(model_complexity=1)
        self._landmarks: list[np.ndarray] = []
        self._pose_detected: list[bool] = []
        self._left_hand_detected: list[bool] = []
        self._right_hand_detected: list[bool] = []
        self._received_frames = 0
        self._latest_result: RecognitionResult | None = None
        self._lock = threading.Lock()
        self._closed = False

    def add_yuv420_frame(
        self,
        payload: bytes,
        *,
        width: int,
        height: int,
        rotation: int,
        plane_lengths: list[int],
        row_strides: list[int],
        pixel_strides: list[int],
    ) -> None:
        if len(plane_lengths) != 3 or len(row_strides) != 3:
            raise SignRecognitionError("相機影格格式不正確")
        if len(payload) != sum(plane_lengths):
            raise SignRecognitionError("相機影格資料長度不正確")

        offsets = np.cumsum([0, *plane_lengths])
        planes = [
            np.frombuffer(payload[offsets[i] : offsets[i + 1]], dtype=np.uint8)
            for i in range(3)
        ]
        y = self._read_plane(planes[0], width, height, row_strides[0], 1)
        chroma_width = (width + 1) // 2
        chroma_height = (height + 1) // 2
        u = self._read_plane(
            planes[1],
            chroma_width,
            chroma_height,
            row_strides[1],
            max(1, pixel_strides[1]),
        )
        v = self._read_plane(
            planes[2],
            chroma_width,
            chroma_height,
            row_strides[2],
            max(1, pixel_strides[2]),
        )
        u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

        yuv = np.stack((y, u, v), axis=2).astype(np.float32)
        yuv[:, :, 1:] -= 128.0
        b = yuv[:, :, 0] + 1.772 * yuv[:, :, 1]
        g = yuv[:, :, 0] - 0.344136 * yuv[:, :, 1] - 0.714136 * yuv[:, :, 2]
        r = yuv[:, :, 0] + 1.402 * yuv[:, :, 2]
        frame = np.clip(np.stack((b, g, r), axis=2), 0, 255).astype(np.uint8)

        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        with self._lock:
            if self._closed:
                raise SignRecognitionError("即時辨識工作階段已結束")
            points = self._extractor(frame)
            self._landmarks.append(points)
            self._pose_detected.append(not np.isnan(points[:12]).all())
            self._left_hand_detected.append(not np.isnan(points[12:33]).all())
            self._right_hand_detected.append(not np.isnan(points[33:54]).all())
            self._received_frames += 1
            if len(self._landmarks) > MAX_LANDMARK_FRAMES:
                self._landmarks.pop(0)
                self._pose_detected.pop(0)
                self._left_hand_detected.pop(0)
                self._right_hand_detected.pop(0)
            pose_frames = sum(self._pose_detected)
            left_hand_frames = sum(self._left_hand_detected)
            right_hand_frames = sum(self._right_hand_detected)
            if (
                len(self._landmarks) >= MIN_INPUT_FRAMES
                and pose_frames >= MIN_INPUT_FRAMES
                and self._received_frames % 4 == 0
            ):
                self._latest_result = self._service.recognize_landmarks(
                    np.asarray(self._landmarks, dtype=np.float32),
                    pose_frames,
                    left_hand_frames,
                    right_hand_frames,
                    require_hands=False,
                )

    @staticmethod
    def _read_plane(
        plane: np.ndarray,
        width: int,
        height: int,
        row_stride: int,
        pixel_stride: int,
    ) -> np.ndarray:
        rows = np.arange(height, dtype=np.int64)[:, None] * row_stride
        columns = np.arange(width, dtype=np.int64)[None, :] * pixel_stride
        indices = rows + columns
        if indices.size and int(indices.max()) >= len(plane):
            raise SignRecognitionError("相機影格平面資料不完整")
        return plane[indices]

    def finish(self) -> RecognitionResult:
        with self._lock:
            if self._closed:
                raise SignRecognitionError("即時辨識工作階段已結束")
            self._closed = True
            self._extractor.close()
            landmarks = np.asarray(self._landmarks, dtype=np.float32)
            detected_pose_frames = sum(self._pose_detected)
            left_hand_frames = sum(self._left_hand_detected)
            right_hand_frames = sum(self._right_hand_detected)
            latest_result = self._latest_result
        if max(left_hand_frames, right_hand_frames) < MIN_INPUT_FRAMES:
            raise SignRecognitionError(
                f"手部關鍵點偵測不足（共 {len(landmarks)} 格，"
                f"左手 {left_hand_frames} 格、右手 {right_hand_frames} 格），"
                "請確認手機畫面方向正確，並讓雙手完整入鏡"
            )
        if latest_result is not None:
            return latest_result
        return self._service.recognize_landmarks(
            landmarks,
            detected_pose_frames,
            left_hand_frames,
            right_hand_frames,
        )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._extractor.close()


class SignRecognitionService:
    """Extract landmarks from one recorded sign and classify it."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None
        self._classes: list[str] = []
        self._seq_len = 64
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        if not CHECKPOINT_PATH.is_file():
            raise SignRecognitionError(f"找不到辨識模型：{CHECKPOINT_PATH}")
        self._display_labels = self._load_display_labels()

    @staticmethod
    def _load_display_labels() -> dict[int, str]:
        if not SIGN_WORDS_PATH.is_file():
            raise SignRecognitionError(f"找不到手語中文映射：{SIGN_WORDS_PATH}")
        try:
            entries = json.loads(SIGN_WORDS_PATH.read_text(encoding="utf-8"))
            labels: dict[int, str] = {}
            for entry in entries:
                number, separator, chinese = str(entry).partition(".")
                if not separator or not number.isdigit() or not chinese.strip():
                    raise ValueError(f"無效項目：{entry!r}")
                labels[int(number)] = chinese.strip()
            return labels
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SignRecognitionError(f"無法讀取手語中文映射：{exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_models(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            try:
                checkpoint = torch.load(
                    CHECKPOINT_PATH,
                    map_location=self.device,
                    weights_only=True,
                )
                apply_feature_flags(checkpoint)
                model = build_from_checkpoint(checkpoint).to(self.device)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
            except Exception as exc:
                raise SignRecognitionError(f"無法載入辨識模型：{exc}") from exc

            self._classes = list(checkpoint["classes"])
            self._seq_len = int(checkpoint.get("seq_len", 64))
            self._model = model

    def recognize_video(self, video_path: Path) -> RecognitionResult:
        self.load_models()
        landmarks, pose_frames, left_hand_frames, right_hand_frames = (
            self._extract_landmarks(video_path)
        )
        return self.recognize_landmarks(
            landmarks,
            pose_frames,
            left_hand_frames,
            right_hand_frames,
        )

    def start_streaming_session(self) -> StreamingRecognitionSession:
        self.load_models()
        return StreamingRecognitionSession(self)

    def recognize_landmarks(
        self,
        landmarks: np.ndarray,
        detected_pose_frames: int | None = None,
        left_hand_frames: int | None = None,
        right_hand_frames: int | None = None,
        *,
        require_hands: bool = True,
    ) -> RecognitionResult:
        frame_count = len(landmarks)
        if frame_count < MIN_INPUT_FRAMES:
            raise SignRecognitionError(
                f"影片影格太少：{frame_count} 格，至少需要 {MIN_INPUT_FRAMES} 格"
            )
        if detected_pose_frames is not None and detected_pose_frames < MIN_INPUT_FRAMES:
            raise SignRecognitionError(
                "影片中偵測到身體姿勢的影格太少，請讓上半身完整入鏡"
            )
        if require_hands and left_hand_frames is not None and right_hand_frames is not None:
            if max(left_hand_frames, right_hand_frames) < MIN_INPUT_FRAMES:
                raise SignRecognitionError(
                    "手部關鍵點偵測不足，請讓雙手完整入鏡後重新錄製"
                )

        predictions = self._predict(landmarks, top_k=3)
        raw_label, confidence = predictions[0]
        class_number = raw_label.partition(".")[0]
        display_label = (
            self._display_labels.get(int(class_number))
            if class_number.isdigit()
            else None
        )
        accepted = confidence >= self.confidence_threshold and display_label is not None
        if display_label is None:
            message = f"辨識類別 {class_number} 尚未設定中文名稱"
        elif confidence < self.confidence_threshold:
            message = "辨識信心度不足，請重新錄製手語"
        else:
            message = None

        candidates = []
        for candidate_raw_label, candidate_confidence in predictions:
            candidate_class_number = candidate_raw_label.partition(".")[0]
            candidate_label = (
                self._display_labels.get(int(candidate_class_number))
                if candidate_class_number.isdigit()
                else None
            )
            if candidate_label is None:
                continue
            candidates.append(
                RecognitionCandidate(
                    label=candidate_label,
                    raw_label=candidate_raw_label,
                    confidence=round(candidate_confidence, 6),
                    confidence_percent=round(candidate_confidence * 100, 1),
                )
            )

        return RecognitionResult(
            label=display_label if accepted else None,
            raw_label=raw_label,
            confidence=round(confidence, 6),
            confidence_percent=round(confidence * 100, 1),
            accepted=accepted,
            frame_count=frame_count,
            candidates=candidates,
            message=message,
            pose_frame_count=detected_pose_frames,
            left_hand_frame_count=left_hand_frames,
            right_hand_frame_count=right_hand_frames,
        )

    @staticmethod
    def _extract_landmarks(video_path: Path) -> tuple[np.ndarray, int, int, int]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise SignRecognitionError("無法解碼上傳的影片")

        sequence: list[np.ndarray] = []
        detected_pose_frames = 0
        left_hand_frames = 0
        right_hand_frames = 0
        extractor: HolisticExtractor | None = None
        total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        sampled_frames = None
        if total_frames > MAX_LANDMARK_FRAMES:
            sampled_frames = set(
                np.linspace(
                    0,
                    total_frames - 1,
                    num=MAX_LANDMARK_FRAMES,
                    dtype=np.int64,
                ).tolist()
            )
        frame_index = 0
        try:
            extractor = HolisticExtractor(model_complexity=1)
            while True:
                success, frame = capture.read()
                if not success:
                    break
                should_process = sampled_frames is None or frame_index in sampled_frames
                frame_index += 1
                if not should_process:
                    continue
                points = extractor(frame)
                if not np.isnan(points[:12]).all():
                    detected_pose_frames += 1
                if not np.isnan(points[12:33]).all():
                    left_hand_frames += 1
                if not np.isnan(points[33:54]).all():
                    right_hand_frames += 1
                sequence.append(points)
        finally:
            capture.release()
            if extractor is not None:
                extractor.close()

        if not sequence:
            raise SignRecognitionError("影片中沒有可處理的影格")
        return (
            np.asarray(sequence, dtype=np.float32),
            detected_pose_frames,
            left_hand_frames,
            right_hand_frames,
        )

    def _predict(
        self, landmarks: np.ndarray, *, top_k: int = 3
    ) -> list[tuple[str, float]]:
        if self._model is None:
            raise SignRecognitionError("辨識模型尚未載入")

        coords, mask = normalize_sequence(landmarks)
        coords = resample_time(coords, self._seq_len)
        mask = resample_time(mask[:, :, None], self._seq_len)[:, :, 0]
        features = build_features(coords, mask)
        tensor = torch.from_numpy(features).unsqueeze(0).float().to(self.device)

        with self._inference_lock, torch.inference_mode():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0]
            count = min(top_k, len(self._classes))
            confidences, predictions = torch.topk(probabilities, k=count)
        return [
            (self._classes[index.item()], confidence.item())
            for confidence, index in zip(confidences, predictions, strict=True)
        ]
