"""將多個共用骨架的 GLB 動畫串成單一連續動畫（含 NumPy 加速與快取支援）。

用法：
    python merge_glb.py --output static/output.glb \
        adjusted_actions/at.glb adjusted_actions/you.glb
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pygltflib import (
    GLTF2,
    Accessor,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    BufferView,
)

# ---------------------------------------------------------------------------
# 各 accessor 類型對應的分量數
# ---------------------------------------------------------------------------
_TYPE_TO_COUNT: dict[str, int] = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


# ---------------------------------------------------------------------------
# NumPy 加速版 accessor 讀寫
# ---------------------------------------------------------------------------


def read_accessor_np(
    gltf: GLTF2,
    accessor_index: int,
    binary_blob: bytes | bytearray,
    component_count: int,
) -> np.ndarray:
    """讀取 GLTF accessor 資料並回傳 float32 NumPy 陣列。

    Returns:
        shape ``(N,)`` 用於 SCALAR，shape ``(N, C)`` 用於 VEC2/VEC3/VEC4。
    """
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    element_size = component_count * 4
    stride = buffer_view.byteStride or element_size

    if stride == element_size:
        # 連續排列 -- 整塊批量讀取
        arr = np.frombuffer(
            binary_blob,
            dtype=np.float32,
            count=accessor.count * component_count,
            offset=offset,
        ).copy()
    else:
        # 跨距排列 -- 逐元素讀取
        arr = np.empty(accessor.count * component_count, dtype=np.float32)
        for i in range(accessor.count):
            src = offset + i * stride
            arr[i * component_count : (i + 1) * component_count] = np.frombuffer(
                binary_blob, dtype=np.float32, count=component_count, offset=src
            )

    if component_count == 1:
        return arr
    return arr.reshape(-1, component_count)


def write_accessor_np(
    gltf: GLTF2,
    binary_blob: bytearray,
    values: np.ndarray,
    type_name: str,
    component_count: int,
) -> int:
    """將 NumPy 陣列打包寫入 binary blob 並註冊新的 accessor。回傳 accessor 索引。"""
    flat = values.astype(np.float32, copy=False).ravel()
    if flat.size == 0:
        raise ValueError("無法建立沒有數值的 accessor")

    chunk = flat.tobytes()

    padding = (-len(binary_blob)) % 4
    if padding:
        binary_blob.extend(b"\x00" * padding)

    offset = len(binary_blob)
    binary_blob.extend(chunk)
    gltf.bufferViews.append(
        BufferView(buffer=0, byteOffset=offset, byteLength=len(chunk))
    )

    if component_count == 1:
        count = flat.size
        minimum = [float(flat.min())]
        maximum = [float(flat.max())]
    else:
        arr2d = flat.reshape(-1, component_count)
        count = arr2d.shape[0]
        minimum = arr2d.min(axis=0).tolist()
        maximum = arr2d.max(axis=0).tolist()

    gltf.accessors.append(
        Accessor(
            bufferView=len(gltf.bufferViews) - 1,
            byteOffset=0,
            componentType=5126,
            count=count,
            type=type_name,
            min=minimum,
            max=maximum,
        )
    )
    return len(gltf.accessors) - 1


# ---------------------------------------------------------------------------
# 動畫輔助函式
# ---------------------------------------------------------------------------


def animation_bounds(
    gltf: GLTF2,
    animation: Animation,
    binary_blob: bytes | bytearray,
) -> tuple[float, float]:
    """回傳動畫的 (起始時間, 結束時間)。"""
    parts: list[np.ndarray] = []
    for channel in animation.channels:
        sampler = animation.samplers[channel.sampler]
        parts.append(read_accessor_np(gltf, sampler.input, binary_blob, 1))
    if not parts:
        raise ValueError("GLB 動畫沒有時間軸資料")
    combined = np.concatenate(parts)
    return float(combined.min()), float(combined.max())


def trim_animation_duration(
    gltf: GLTF2,
    binary_blob: bytearray,
    max_duration_seconds: float,
) -> None:
    """保留動畫原速，只裁切超過指定長度的時間軸。"""
    if max_duration_seconds <= 0:
        raise ValueError("動畫長度必須大於 0 秒")

    animation = gltf.animations[0]
    start, end = animation_bounds(gltf, animation, binary_blob)
    duration = end - start
    if duration <= max_duration_seconds:
        return

    cutoff = start + max_duration_seconds
    for sampler in animation.samplers:
        interpolation = sampler.interpolation or "LINEAR"
        if interpolation not in {"LINEAR", "STEP"}:
            raise ValueError(f"暫不支援裁切 {interpolation} 插值的動畫")

        output_accessor = gltf.accessors[sampler.output]
        component_count = _TYPE_TO_COUNT.get(output_accessor.type)
        if component_count is None:
            raise ValueError(f"不支援的動畫輸出類型：{output_accessor.type}")

        times = read_accessor_np(gltf, sampler.input, binary_blob, 1)
        values = read_accessor_np(gltf, sampler.output, binary_blob, component_count)

        keep_count = int(np.sum(times <= cutoff + 1e-6))
        if keep_count == 0:
            keep_count = 1

        trimmed_times = times[:keep_count].copy()
        trimmed_values = values[:keep_count].copy()

        last_time = float(trimmed_times[-1])
        if last_time < cutoff - 1e-6 and keep_count < len(times):
            prev_time = float(times[keep_count - 1])
            next_time = float(times[keep_count])
            prev_val = values[keep_count - 1]
            next_val = values[keep_count]

            if interpolation == "LINEAR":
                ratio = (cutoff - prev_time) / (next_time - prev_time)
                if component_count == 1:
                    cutoff_val = float(prev_val + (next_val - prev_val) * ratio)
                    trimmed_times = np.append(trimmed_times, cutoff)
                    trimmed_values = np.append(trimmed_values, cutoff_val)
                else:
                    pa = prev_val.copy()
                    na = next_val.copy()
                    if component_count == 4 and float(np.dot(pa, na)) < 0:
                        na = -na
                    interp = pa + (na - pa) * ratio
                    if component_count == 4:
                        norm = np.linalg.norm(interp)
                        if norm > 0:
                            interp = interp / norm
                    trimmed_times = np.append(trimmed_times, cutoff)
                    trimmed_values = np.vstack(
                        [trimmed_values, interp.reshape(1, -1)]
                    )
            else:
                trimmed_times = np.append(trimmed_times, cutoff)
                if component_count == 1:
                    trimmed_values = np.append(trimmed_values, float(prev_val))
                else:
                    trimmed_values = np.vstack(
                        [trimmed_values, prev_val.reshape(1, -1)]
                    )
        elif abs(last_time - cutoff) <= 1e-6:
            trimmed_times[-1] = cutoff

        normalized_times = trimmed_times - start
        sampler.input = write_accessor_np(
            gltf, binary_blob, normalized_times, "SCALAR", 1
        )
        sampler.output = write_accessor_np(
            gltf, binary_blob, trimmed_values, output_accessor.type, component_count
        )


# ---------------------------------------------------------------------------
# 預快取動畫資料結構
# ---------------------------------------------------------------------------


@dataclass
class CachedAnimation:
    """預先擷取的骨架動畫關鍵影格資料。

    每個 GLB 動作檔約 14 MB，但實際的動畫關鍵影格僅 50-200 KB。
    此結構僅保留關鍵影格資料，適合常駐記憶體（88 個檔案共約 18 MB）。
    """

    channel_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, int]] = field(
        default_factory=dict
    )
    # key: (node_name, path)
    # value: (times [shape (N,)], values [shape (N,) 或 (N,C)], component_count)
    start: float = 0.0
    end: float = 0.0


def extract_animation_cache(gltf_path: Path) -> CachedAnimation:
    """載入 GLB 檔案並僅擷取骨架動畫關鍵影格資料。

    回傳的物件通常僅佔 50-200 KB（對比完整 GLB 的 14 MB），
    可安全地將所有 88 個動作檔快取於記憶體中（總計約 18 MB）。
    """
    gltf = GLTF2().load(str(gltf_path))
    blob = gltf.binary_blob()
    if not gltf.animations:
        raise ValueError(f"GLB 缺少動畫：{gltf_path}")

    anim = gltf.animations[0]
    channel_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, int]] = {}
    all_times: list[np.ndarray] = []

    for channel in anim.channels:
        sampler = anim.samplers[channel.sampler]
        node_name = gltf.nodes[channel.target.node].name
        path = channel.target.path

        # 從 accessor 類型推斷分量數，更健壯
        output_accessor = gltf.accessors[sampler.output]
        component_count = _TYPE_TO_COUNT.get(
            output_accessor.type, 4 if path == "rotation" else 3
        )

        times = read_accessor_np(gltf, sampler.input, blob, 1)
        values = read_accessor_np(gltf, sampler.output, blob, component_count)

        channel_data[(node_name, path)] = (times, values, component_count)
        all_times.append(times)

    combined = np.concatenate(all_times) if all_times else np.array([0.0])
    return CachedAnimation(
        channel_data=channel_data,
        start=float(combined.min()),
        end=float(combined.max()),
    )


# ---------------------------------------------------------------------------
# 動畫附加（快取版本，零檔案 I/O）
# ---------------------------------------------------------------------------


def append_animation_cached(
    base: GLTF2,
    base_blob: bytearray,
    source_cache: CachedAnimation,
    transition_seconds: float,
    sequence_name: str,
) -> None:
    """從預快取的關鍵影格資料附加動畫（無檔案 I/O）。"""
    if not base.animations:
        raise ValueError("Base GLB 缺少動畫")

    base_animation = base.animations[0]
    _, base_end = animation_bounds(base, base_animation, base_blob)
    time_shift = base_end + transition_seconds - source_cache.start

    base_node_indexes = {
        node.name: idx for idx, node in enumerate(base.nodes) if node.name
    }
    base_targets = {
        (base.nodes[ch.target.node].name, ch.target.path): ch
        for ch in base_animation.channels
    }

    common_targets = base_targets.keys() & source_cache.channel_data.keys()
    if not common_targets:
        raise ValueError("兩個 GLB 沒有可對應的骨架頻道")

    new_samplers: list[AnimationSampler] = []
    new_channels: list[AnimationChannel] = []

    for node_name, path in sorted(common_targets):
        base_channel = base_targets[(node_name, path)]
        base_sampler = base_animation.samplers[base_channel.sampler]
        source_times, source_values, component_count = source_cache.channel_data[
            (node_name, path)
        ]

        base_times = read_accessor_np(base, base_sampler.input, base_blob, 1)
        base_values = read_accessor_np(
            base, base_sampler.output, base_blob, component_count
        )

        # 確保向量值為 2D 以利 concatenate
        if component_count > 1:
            bv = base_values.reshape(-1, component_count)
            sv = source_values.reshape(-1, component_count)
        else:
            bv = base_values
            sv = source_values

        parts_t: list[np.ndarray] = [base_times]
        parts_v: list[np.ndarray] = [bv]

        if float(base_times[-1]) < base_end:
            parts_t.append(np.array([base_end], dtype=np.float32))
            parts_v.append(bv[-1:])

        parts_t.append(source_times + time_shift)
        parts_v.append(sv)

        merged_times = np.concatenate(parts_t)
        merged_values = np.concatenate(parts_v)

        input_index = write_accessor_np(base, base_blob, merged_times, "SCALAR", 1)
        output_index = write_accessor_np(
            base,
            base_blob,
            merged_values,
            "VEC4" if path == "rotation" else "VEC3",
            component_count,
        )

        new_samplers.append(
            AnimationSampler(
                input=input_index, output=output_index, interpolation="LINEAR"
            )
        )
        new_channels.append(
            AnimationChannel(
                sampler=len(new_samplers) - 1,
                target=AnimationChannelTarget(
                    node=base_node_indexes[node_name], path=path
                ),
            )
        )

    base.animations = [
        Animation(name=sequence_name, samplers=new_samplers, channels=new_channels)
    ]


def append_animation(
    base: GLTF2,
    base_blob: bytearray,
    source_path: Path,
    transition_seconds: float,
    sequence_name: str,
) -> None:
    """從磁碟載入來源 GLB 並附加動畫（向後相容介面）。"""
    cache = extract_animation_cache(source_path)
    append_animation_cached(base, base_blob, cache, transition_seconds, sequence_name)


# ---------------------------------------------------------------------------
# 時間軸計算（純記憶體，無 I/O）
# ---------------------------------------------------------------------------


def compute_timeline(
    action_caches: list[CachedAnimation],
    action_words: list[str],
    first_duration_seconds: float = 0.8,
    transition_seconds: float = 0.8,
) -> list[dict[str, str | float]]:
    """從快取資料計算動畫時間軸，完全不需檔案 I/O。"""
    timeline: list[dict[str, str | float]] = []
    cursor = first_duration_seconds
    for word, cache in zip(action_words, action_caches):
        action_duration = cache.end - cache.start
        word_start = cursor
        word_end = cursor + transition_seconds + action_duration
        timeline.append({"word": word, "start": word_start, "end": word_end})
        cursor = word_end
    return timeline


# ---------------------------------------------------------------------------
# 高階合併（快取版本，含時間軸，單次流程）
# ---------------------------------------------------------------------------


def merge_glb_with_timeline(
    idle_path: Path,
    action_caches: list[CachedAnimation],
    action_words: list[str],
    idle_cache: CachedAnimation,
    output_path: Path,
    transition_seconds: float = 0.8,
    first_duration_seconds: float = 0.8,
    first_transition_seconds: float = 0.8,
) -> list[dict[str, str | float]]:
    """合併動畫並在單次流程中計算時間軸。

    僅從磁碟載入 idle GLB（取得基底模型網格）；所有動作
    關鍵影格皆來自預快取的 ``CachedAnimation`` 物件。

    回傳動畫時間軸。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = GLTF2().load(str(idle_path))
    if not base.animations:
        raise ValueError(f"基底 GLB 缺少動畫：{idle_path}")
    base_blob = bytearray(base.binary_blob())
    trim_animation_duration(base, base_blob, first_duration_seconds)

    timeline: list[dict[str, str | float]] = []
    cursor = first_duration_seconds
    sequence_names = [idle_path.stem]

    for i, (cache, word) in enumerate(zip(action_caches, action_words)):
        trans = first_transition_seconds if i == 0 else transition_seconds

        action_duration = cache.end - cache.start
        word_start = cursor
        word_end = cursor + trans + action_duration
        timeline.append({"word": word, "start": word_start, "end": word_end})
        cursor = word_end

        sequence_names.append(word)
        append_animation_cached(
            base, base_blob, cache, trans, "_then_".join(sequence_names)
        )

    # 附加結尾 idle 動畫
    sequence_names.append(idle_path.stem)
    append_animation_cached(
        base, base_blob, idle_cache, transition_seconds, "_then_".join(sequence_names)
    )

    base.buffers[0].byteLength = len(base_blob)
    base.set_binary_blob(bytes(base_blob))
    for accessor in base.accessors:
        if accessor.bufferView is None and accessor.byteOffset is not None:
            accessor.byteOffset = None
    base.save(str(output_path))

    return timeline


# ---------------------------------------------------------------------------
# 原始檔案式合併（CLI 與向後相容）
# ---------------------------------------------------------------------------


def merge_glb_files(
    input_paths: list[Path],
    output_path: Path,
    transition_seconds: float = 0.5,
    first_duration_seconds: float | None = None,
    first_transition_seconds: float | None = None,
) -> Path:
    if not input_paths:
        raise ValueError("至少需要一個 GLB 輸入檔")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"找不到 GLB：{path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        shutil.copyfile(input_paths[0], output_path)
        return output_path

    base = GLTF2().load(str(input_paths[0]))
    if not base.animations:
        raise ValueError(f"基底 GLB 缺少動畫：{input_paths[0]}")
    base_blob = bytearray(base.binary_blob())
    if first_duration_seconds is not None:
        trim_animation_duration(base, base_blob, first_duration_seconds)
    sequence_names = [input_paths[0].stem]

    for source_index, source_path in enumerate(input_paths[1:], start=1):
        sequence_names.append(source_path.stem)
        append_animation(
            base,
            base_blob,
            source_path,
            first_transition_seconds
            if source_index == 1 and first_transition_seconds is not None
            else transition_seconds,
            "_then_".join(sequence_names),
        )

    base.buffers[0].byteLength = len(base_blob)
    base.set_binary_blob(bytes(base_blob))
    for accessor in base.accessors:
        if accessor.bufferView is None and accessor.byteOffset is not None:
            accessor.byteOffset = None
    base.save(str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="依序輸入的 GLB")
    parser.add_argument("--output", required=True, type=Path, help="輸出 GLB")
    parser.add_argument(
        "--transition",
        type=float,
        default=0.5,
        help="動作間過渡秒數，預設 0.5",
    )
    parser.add_argument(
        "--first-duration",
        type=float,
        default=None,
        help="保留原速，將第一段動畫裁切至指定秒數",
    )
    parser.add_argument(
        "--first-transition",
        type=float,
        default=None,
        help="第一段與第二段之間的過渡秒數",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = merge_glb_files(
        args.inputs,
        args.output,
        args.transition,
        args.first_duration,
        args.first_transition,
    )
    print(f"輸出完成：{output}")


if __name__ == "__main__":
    main()
