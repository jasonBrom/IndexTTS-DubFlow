from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np

from .models import Segment
from .utils import CommandError, require_binary, run_command, safe_name, stable_hash

_MIX_SAMPLE_RATE = 48_000
_LIMITER_ATTACK_MS = 5.0


@lru_cache(maxsize=1)
def _alimiter_supports_latency() -> bool:
    """Return whether this FFmpeg build exposes alimiter's latency option."""
    try:
        result = run_command(
            ["ffmpeg", "-hide_banner", "-h", "filter=alimiter"],
        )
    except CommandError:
        return False
    return any(
        line.strip().startswith("latency ")
        for line in (result.stdout or "").splitlines()
    )


def _mix_filter_graph(background_volume: float, dub_volume: float) -> str:
    common = (
        f"[0:a]volume={background_volume:.4f}[bg];"
        f"[1:a]volume={dub_volume:.4f}[dub];"
        "[bg][dub]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
        f"aresample={_MIX_SAMPLE_RATE},"
    )
    if _alimiter_supports_latency():
        return common + f"alimiter=limit=0.98:attack={_LIMITER_ATTACK_MS:g}:latency=1[a]"

    # Older Colab FFmpeg builds have alimiter but not its `latency` switch.
    # alimiter delays output by buffer_size - 1 samples. Trim that lookahead
    # delay and pad the tail so the mixed track keeps both alignment and length.
    delay_samples = max(
        0,
        round(_MIX_SAMPLE_RATE * _LIMITER_ATTACK_MS / 1000) - 1,
    )
    return (
        common
        + f"alimiter=limit=0.98:attack={_LIMITER_ATTACK_MS:g},"
        + f"atrim=start_sample={delay_samples},asetpts=PTS-STARTPTS,"
        + f"apad=pad_len={delay_samples}[a]"
    )


def probe_media(path: str | Path) -> dict:
    require_binary("ffprobe")
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ]
    )
    value = json.loads(result.stdout or "{}")
    try:
        value["duration"] = float(value["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"无法读取媒体时长：{path}") from exc
    return value


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> Path:
    require_binary("ffmpeg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return output_path


def separate_vocals(
    audio_path: str | Path,
    output_root: str | Path,
    *,
    device: str = "cuda",
    model: str = "htdemucs",
) -> tuple[Path, Path]:
    """Return (vocals, background) from Demucs' two-stem mode."""
    output_root = Path(output_root)
    stem_dir = output_root / model / Path(audio_path).stem
    vocals = stem_dir / "vocals.wav"
    background = stem_dir / "no_vocals.wav"
    if vocals.is_file() and background.is_file():
        return vocals, background
    try:
        __import__("demucs")
    except ImportError as exc:
        raise RuntimeError(
            "当前环境没有 Demucs。请重新运行安装单元，或关闭“分离人声/背景”。"
        ) from exc
    run_command(
        [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-n",
            model,
            "--device",
            device,
            "-o",
            str(output_root),
            str(audio_path),
        ],
        capture=False,
    )
    if not vocals.is_file() or not background.is_file():
        raise RuntimeError("Demucs 已运行，但没有生成预期的 vocals/no_vocals 文件。")
    return vocals, background


def _read_audio(path: str | Path, sample_rate: int, mono: bool = True) -> np.ndarray:
    import librosa

    data, _ = librosa.load(str(path), sr=sample_rate, mono=mono)
    return np.asarray(data, dtype=np.float32)


def _write_audio(path: str | Path, data: np.ndarray, sample_rate: int) -> Path:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate, subtype="PCM_16")
    return path


def build_speaker_references(
    source_audio: str | Path,
    segments: list[Segment],
    output_dir: str | Path,
    *,
    sample_rate: int = 16_000,
    max_seconds: float = 12.0,
    min_clip_seconds: float = 0.8,
) -> dict[str, Path]:
    """Build stable per-speaker timbre prompts from the longest labelled clips."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _read_audio(source_audio, sample_rate)
    silence = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
    by_speaker: dict[str, list[Segment]] = {}
    for segment in segments:
        if segment.enabled:
            by_speaker.setdefault(segment.speaker or "SPEAKER_00", []).append(segment)

    references: dict[str, Path] = {}
    max_samples = int(max_seconds * sample_rate)
    for speaker, speaker_segments in by_speaker.items():
        clips: list[np.ndarray] = []
        used = 0
        ranked = sorted(speaker_segments, key=lambda item: item.duration, reverse=True)
        for segment in ranked:
            start = max(0, int(round(segment.start * sample_rate)))
            end = min(len(source), int(round(segment.end * sample_rate)))
            clip = source[start:end]
            if len(clip) < int(min_clip_seconds * sample_rate):
                continue
            edge = min(int(0.06 * sample_rate), len(clip) // 10)
            if edge:
                clip = clip[edge:-edge]
            remaining = max_samples - used
            if remaining <= int(0.5 * sample_rate):
                break
            clip = clip[:remaining]
            clips.append(clip)
            used += len(clip)
            if used < max_samples:
                clips.append(silence)
                used += len(silence)
        if not clips:
            first = ranked[0]
            center = (first.start + first.end) / 2
            start_s = max(0.0, center - 1.5)
            end_s = min(len(source) / sample_rate, center + 1.5)
            clips = [source[int(start_s * sample_rate) : int(end_s * sample_rate)]]
        combined = np.concatenate(clips)[:max_samples]
        peak = float(np.max(np.abs(combined))) if combined.size else 0.0
        if peak > 0.98:
            combined = combined * (0.98 / peak)
        ref_path = output_dir / (
            f"{safe_name(speaker, 'speaker')}_{stable_hash([speaker])[:8]}.wav"
        )
        _write_audio(ref_path, combined, sample_rate)
        references[speaker] = ref_path
    return references


def build_emotion_references(
    source_audio: str | Path,
    segments: list[Segment],
    output_dir: str | Path,
    *,
    sample_rate: int = 16_000,
    min_seconds: float = 1.2,
    max_seconds: float = 6.0,
) -> dict[int, Path]:
    """Build clean per-line emotion prompts without leaking adjacent speakers.

    The current utterance always comes first.  When it is too short for a stable
    emotion embedding, only clips carrying the same speaker label are appended;
    we never enlarge the raw time window into a neighbouring speaker.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _read_audio(source_audio, sample_rate)
    result: dict[int, Path] = {}
    by_speaker: dict[str, list[Segment]] = {}
    for item in segments:
        if item.enabled:
            by_speaker.setdefault(item.speaker or "SPEAKER_00", []).append(item)

    def exact_clip(item: Segment) -> np.ndarray:
        start = max(0, int(round(item.start * sample_rate)))
        end = min(len(source), int(round(item.end * sample_rate)))
        clip = source[start:end]
        edge = min(int(0.025 * sample_rate), len(clip) // 12)
        return clip[edge:-edge] if edge and len(clip) > edge * 2 else clip

    for segment in segments:
        clips = [exact_clip(segment)]
        current_samples = len(clips[0])
        minimum_samples = int(min_seconds * sample_rate)
        maximum_samples = int(max_seconds * sample_rate)
        if current_samples < minimum_samples:
            same_speaker = sorted(
                (
                    item
                    for item in by_speaker.get(segment.speaker or "SPEAKER_00", [])
                    if item.id != segment.id
                ),
                key=lambda item: (
                    abs(((item.start + item.end) / 2) - ((segment.start + segment.end) / 2)),
                    -item.duration,
                ),
            )
            separator = np.zeros(int(0.08 * sample_rate), dtype=np.float32)
            for candidate in same_speaker:
                if current_samples >= minimum_samples or current_samples >= maximum_samples:
                    break
                candidate_clip = exact_clip(candidate)
                if len(candidate_clip) < int(0.35 * sample_rate):
                    continue
                remaining = maximum_samples - current_samples
                addition = np.concatenate([separator, candidate_clip])[:remaining]
                clips.append(addition)
                current_samples += len(addition)
        clip = np.concatenate(clips)[:maximum_samples]
        if len(clip) < minimum_samples:
            clip = np.pad(clip, (0, minimum_samples - len(clip)))
        path = output_dir / f"segment_{segment.id:05d}.wav"
        _write_audio(path, clip, sample_rate)
        result[segment.id] = path
    return result


def build_dialogue_removed_background(
    instrumental_path: str | Path,
    full_vocals_path: str | Path,
    dialogue_segments: Iterable[Segment],
    output_path: str | Path,
    *,
    margin_ms: int = 100,
    fade_ms: int = 45,
) -> Path:
    """Recombine Demucs stems while muting vocals only around enabled dialogue.

    Song vocals and other non-dialogue vocal content remain in the background. If
    dialogue and singing overlap in the same stem/time window, both are attenuated;
    a two-stem model cannot untangle that case.
    """
    import soundfile as sf

    instrumental, sample_rate = sf.read(str(instrumental_path), dtype="float32", always_2d=True)
    vocals, vocal_rate = sf.read(str(full_vocals_path), dtype="float32", always_2d=True)
    if vocal_rate != sample_rate:
        import librosa

        vocals = librosa.resample(vocals.T, orig_sr=vocal_rate, target_sr=sample_rate, axis=-1).T
    channels = max(instrumental.shape[1], vocals.shape[1])

    def match_channels(audio: np.ndarray, wanted: int) -> np.ndarray:
        if audio.shape[1] == wanted:
            return audio
        if audio.shape[1] == 1:
            return np.repeat(audio, wanted, axis=1)
        repeats = math.ceil(wanted / audio.shape[1])
        return np.tile(audio, (1, repeats))[:, :wanted]

    instrumental = match_channels(instrumental, channels)
    vocals = match_channels(vocals, channels)
    length = max(len(instrumental), len(vocals))
    if len(instrumental) < length:
        instrumental = np.pad(instrumental, ((0, length - len(instrumental)), (0, 0)))
    if len(vocals) < length:
        vocals = np.pad(vocals, ((0, length - len(vocals)), (0, 0)))

    keep = np.ones(length, dtype=np.float32)
    margin = int(max(0, margin_ms) * sample_rate / 1000)
    fade = int(max(0, fade_ms) * sample_rate / 1000)
    for segment in dialogue_segments:
        if not segment.enabled:
            continue
        start = max(0, int(round(segment.start * sample_rate)) - margin)
        end = min(length, int(round(segment.end * sample_rate)) + margin)
        if end <= start:
            continue
        keep[start:end] = 0.0
        if fade:
            left_start = max(0, start - fade)
            left = np.linspace(1.0, 0.0, start - left_start, endpoint=False, dtype=np.float32)
            keep[left_start:start] = np.minimum(keep[left_start:start], left)
            right_end = min(length, end + fade)
            right = np.linspace(0.0, 1.0, right_end - end, endpoint=False, dtype=np.float32)
            keep[end:right_end] = np.minimum(keep[end:right_end], right)

    combined = instrumental + vocals * keep[:, None]
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0.99:
        combined *= 0.99 / peak
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), combined, sample_rate, subtype="PCM_16")
    return output_path


def assemble_vocal_track(
    segments: Iterable[Segment],
    output_path: str | Path,
    *,
    total_duration: float,
    sample_rate: int = 22_050,
    gain: float = 1.0,
) -> Path:
    length = max(1, int(math.ceil(total_duration * sample_rate)))
    track = np.zeros(length, dtype=np.float32)
    for segment in segments:
        if not segment.enabled or not segment.generated_path:
            continue
        audio = _read_audio(segment.generated_path, sample_rate)
        render_start = segment.start if segment.render_start is None else segment.render_start
        start = int(round(render_start * sample_rate))
        expected = max(1, int(round(segment.rendered_duration * sample_rate)))
        if len(audio) < expected:
            audio = np.pad(audio, (0, expected - len(audio)))
        else:
            audio = audio[:expected]
        fade = min(int(0.01 * sample_rate), len(audio) // 2)
        if fade:
            audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        end = min(length, start + len(audio))
        if end > start:
            track[start:end] += audio[: end - start] * float(gain)
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.98:
        track *= 0.98 / peak
    return _write_audio(output_path, track, sample_rate)


def mix_audio(
    background_path: str | Path,
    dubbed_vocals_path: str | Path,
    output_path: str | Path,
    *,
    background_volume: float = 1.0,
    dub_volume: float = 1.0,
) -> Path:
    require_binary("ffmpeg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    supports_native_latency = _alimiter_supports_latency()
    if supports_native_latency:
        print("FFmpeg 混音：使用 alimiter 原生延迟补偿。", flush=True)
    else:
        print(
            "FFmpeg 混音：当前 alimiter 不支持 latency 选项，已启用兼容的 5 ms 手动延迟补偿。",
            flush=True,
        )
    filter_graph = _mix_filter_graph(background_volume, dub_volume)
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(background_path),
            "-i",
            str(dubbed_vocals_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[a]",
            "-ar",
            str(_MIX_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return output_path


def mux_video(
    source_video: str | Path,
    mixed_audio: str | Path,
    output_video: str | Path,
    *,
    subtitle_path: str | Path | None = None,
    subtitle_language: str = "und",
    subtitle_mode: str = "soft",
) -> Path:
    require_binary("ffmpeg")
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if subtitle_path and subtitle_mode == "burn":
        escaped = str(Path(subtitle_path).resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_video), "-i", str(mixed_audio),
            "-vf", f"subtitles='{escaped}'",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-metadata", "comment=AI-generated dubbing with IndexTTS 2.5",
            "-shortest", "-movflags", "+faststart",
            str(output_video),
        ]
    elif subtitle_path and subtitle_mode == "soft":
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_video), "-i", str(mixed_audio), "-i", str(subtitle_path),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
            "-metadata:s:s:0", f"language={subtitle_language}",
            "-metadata", "comment=AI-generated dubbing with IndexTTS 2.5",
            "-movflags", "+faststart", str(output_video),
        ]
    else:
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_video), "-i", str(mixed_audio),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-metadata", "comment=AI-generated dubbing with IndexTTS 2.5",
            "-shortest", "-movflags", "+faststart",
            str(output_video),
        ]
    try:
        run_command(args)
    except CommandError:
        if subtitle_mode == "burn":
            raise
        # Some source codecs/containers (for example VP8 WebM) cannot be copied into MP4.
        # Retry with a broadly compatible H.264 encode while preserving all other mappings.
        copy_index = args.index("copy")
        args[copy_index : copy_index + 1] = ["libx264", "-preset", "medium", "-crf", "18"]
        run_command(args)
    return output_video


def scan_drive_videos(
    root: str | Path = "/content/drive/MyDrive",
    limit: int = 500,
    max_entries: int = 10_000,
) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        return []
    extensions = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".ts"}
    values: list[str] = []
    for entry_index, path in enumerate(root.rglob("*")):
        if entry_index >= max_entries:
            break
        if path.is_file() and path.suffix.lower() in extensions:
            values.append(str(path))
            if len(values) >= limit:
                break
    return sorted(values, key=str.lower)
