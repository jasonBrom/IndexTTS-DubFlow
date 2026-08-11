from __future__ import annotations

import inspect
import os
import random
import re
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .indextts_config import inspect_indextts25_config
from .models import DubConfig, Segment
from .utils import file_fingerprint, release_cuda, run_command, stable_hash

MODEL_REQUIRED_FILES = (
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
    "multilingual_zh_ja_yue_char_del.tiktoken",
)

EMOTION_TAG = re.compile(
    r"^\s*(?:\[\s*(?:emotion|情感)\s*[:：]\s*([^\]]+)\]|"
    r"【\s*(?:emotion|情感)\s*[:：]\s*([^】]+)】)\s*",
    re.IGNORECASE,
)


def extract_emotion_tag(text: str) -> tuple[str, str]:
    """Return (spoken text, emotion description) for our review-timeline tag."""
    match = EMOTION_TAG.match(text or "")
    if not match:
        return text.strip(), ""
    emotion = (match.group(1) or match.group(2) or "").strip()
    return text[match.end() :].strip(), emotion


def validate_indextts_install(indextts_dir: str | Path, model_dir: str | Path, config_path: str | Path) -> None:
    indextts_dir = Path(indextts_dir)
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    problems: list[str] = []
    if not (indextts_dir / "indextts" / "infer_v2_5.py").is_file():
        problems.append(f"没有找到 2.5 推理代码：{indextts_dir}")
    if not config_path.is_file():
        problems.append(f"没有找到正确的 2.5 配置：{config_path}")
    missing = [name for name in MODEL_REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing:
        problems.append("模型目录缺少：" + ", ".join(missing))
    if not (model_dir / "qwen0.6bemo4-merge" / "config.json").is_file():
        problems.append("模型目录缺少 QwenEmotion：qwen0.6bemo4-merge/config.json")
    if config_path.is_file():
        try:
            report = inspect_indextts25_config(config_path)
        except ValueError as exc:
            problems.append(str(exc))
        else:
            problems.extend(str(item) for item in report["errors"])
    if problems:
        raise RuntimeError("IndexTTS 2.5 自检失败：\n- " + "\n- ".join(problems))


def _audio_duration(path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


def _atempo_chain(speed: float) -> list[float]:
    if speed <= 0:
        raise ValueError("speed must be positive")
    values: list[float] = []
    while speed > 2.0:
        values.append(2.0)
        speed /= 2.0
    while speed < 0.5:
        values.append(0.5)
        speed /= 0.5
    values.append(speed)
    return values


def fit_audio_exact(path: str | Path, target_duration: float) -> None:
    """Pitch-preserving fallback, followed by exact trim/pad."""
    path = Path(path)
    import soundfile as sf

    info = sf.info(str(path))
    current = info.frames / info.samplerate
    if current <= 0 or target_duration <= 0:
        raise ValueError("audio durations must be positive")
    target_samples = max(1, int(round(target_duration * info.samplerate)))
    if info.frames == target_samples:
        return
    speed = current / target_duration
    filters = [f"atempo={value:.9f}" for value in _atempo_chain(speed)]
    filters.extend([f"apad=pad_dur={target_duration:.9f}", f"atrim=duration={target_duration:.9f}"])
    tmp = path.with_name(path.stem + ".fitted" + path.suffix)
    run_command(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
            "-af", ",".join(filters), "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le", str(tmp),
        ]
    )
    tmp.replace(path)
    verified = sf.info(str(path))
    verified_target = max(1, int(round(target_duration * verified.samplerate)))
    if verified.frames != verified_target:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if len(audio) < verified_target:
            import numpy as np

            audio = np.pad(audio, ((0, verified_target - len(audio)), (0, 0)))
        else:
            audio = audio[:verified_target]
        sf.write(str(path), audio, sample_rate, subtype="PCM_16")


@contextmanager
def _working_directory(path: str | Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class IndexTTS25Engine:
    """One-GPU adapter that adds absolute-duration synthesis to the current 2.5 branch."""

    def __init__(self, config: DubConfig) -> None:
        self.config = config
        self._model = None
        self._lock = threading.Lock()
        self._patched_exact_duration = False

    def load(self, progress: Callable[[str], None] | None = None) -> None:
        if self._model is not None:
            return
        if progress:
            progress("校验 IndexTTS 2.5 代码、配置与主模型文件")
        validate_indextts_install(self.config.indextts_dir, self.config.model_dir, self.config.config_path)
        if progress:
            progress("导入 IndexTTS 2.5 推理模块")
        import torch

        use_bf16 = bool(
            self.config.use_bf16
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        root = str(Path(self.config.indextts_dir).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        with _working_directory(root):
            from indextts.infer_v2_5 import IndexTTS2

            if progress:
                progress("加载 IndexTTS 2.5 checkpoints 到 GPU（首次可能需要几分钟）")
            self._model = IndexTTS2(
                cfg_path=str(Path(self.config.config_path).resolve()),
                model_dir=str(Path(self.config.model_dir).resolve()),
                use_bf16=use_bf16,
                use_cuda_kernel=self.config.use_cuda_kernel,
                use_deepspeed=False,
                use_accel=False,
                use_torch_compile=False,
                use_qwen_emo=self.config.enable_qwen_emotion,
            )
        self._patched_exact_duration = (
            "target_duration_seconds" in inspect.signature(self._model.infer).parameters
        )
        if progress:
            progress("IndexTTS 2.5 主模型已加载")

    @staticmethod
    def _seed(seed: int) -> None:
        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _infer(
        self,
        *,
        reference: str,
        emotion_reference: str | None,
        emotion_text: str | None,
        text: str,
        language: str,
        output_path: str,
        duration_factor: float = 1.0,
        target_duration: float | None = None,
        seed: int,
    ) -> Any:
        self._seed(seed)
        use_text_emotion = bool(emotion_text and emotion_text.strip())
        kwargs: dict[str, Any] = dict(
            spk_audio_prompt=reference,
            emo_audio_prompt=None if use_text_emotion else (emotion_reference or reference),
            emo_alpha=self.config.emotion_strength,
            use_emo_text=use_text_emotion,
            emo_text=emotion_text.strip() if use_text_emotion else None,
            text=text,
            lang=language,
            output_path=output_path,
            interval_silence=0,
            duration_factor=duration_factor,
            do_sample=False,
            num_beams=3,
            max_text_tokens_per_segment=580,
            max_mel_tokens=1810,
            verbose=False,
        )
        if target_duration is not None and self._patched_exact_duration:
            kwargs["target_duration_seconds"] = target_duration
        with _working_directory(self.config.indextts_dir):
            return self._model.infer(**kwargs)

    def synthesize_segment(self, segment: Segment, output_path: str | Path) -> Segment:
        if not segment.target_text.strip():
            raise ValueError(f"片段 {segment.id} 没有译文")
        if not segment.reference_path:
            raise ValueError(f"片段 {segment.id} 没有音色参考")
        self.load()
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spoken_text, inline_emotion = extract_emotion_tag(segment.target_text)
        if not spoken_text:
            raise ValueError(f"片段 {segment.id} 去除情感标签后没有可朗读文本")
        emotion_text = inline_emotion or segment.emotion_text.strip()
        if self.config.emotion_mode == "text" and not emotion_text:
            emotion_text = spoken_text
        if emotion_text and not self.config.enable_qwen_emotion:
            raise RuntimeError("当前片段需要情感描述文本，但 QwenEmotion 未启用")
        max_duration = max(segment.duration, segment.available_duration)
        seed = self.config.seed + segment.id
        if segment.warning.startswith(("时长倍率", "自然语音需")):
            segment.warning = ""
        with self._lock:
            natural_path = output_path.with_name(output_path.stem + ".natural.wav")
            self._infer(
                reference=segment.reference_path,
                emotion_reference=(
                    None if self.config.emotion_mode == "speaker" else segment.emotion_reference_path or None
                ),
                emotion_text=emotion_text or None,
                text=spoken_text,
                language=segment.target_language or self.config.target_language,
                output_path=str(natural_path),
                seed=seed,
            )
            natural_duration = _audio_duration(natural_path)
            segment.natural_duration = natural_duration
            if natural_duration <= max_duration + 0.015:
                natural_path.replace(output_path)
                segment.duration_factor = 1.0
                segment.render_start = segment.start
                segment.render_end = min(
                    segment.start + natural_duration,
                    segment.max_render_end if segment.max_render_end is not None else segment.start + natural_duration,
                )
            elif self._patched_exact_duration:
                self._infer(
                    reference=segment.reference_path,
                    emotion_reference=(
                        None if self.config.emotion_mode == "speaker" else segment.emotion_reference_path or None
                    ),
                    emotion_text=emotion_text or None,
                    text=spoken_text,
                    language=segment.target_language or self.config.target_language,
                    output_path=str(output_path),
                    target_duration=max_duration,
                    seed=seed,
                )
                control = self._model.last_duration_control or {}
                segment.natural_duration = control.get("natural_duration_seconds", natural_duration)
                segment.duration_factor = max_duration / max(0.01, natural_duration)
                segment.render_start = segment.start
                segment.render_end = segment.start + max_duration
                natural_path.unlink(missing_ok=True)
                fit_audio_exact(output_path, max_duration)
            else:
                natural_path.replace(output_path)
                fit_audio_exact(output_path, max_duration)
                natural_path.unlink(missing_ok=True)
                segment.duration_factor = max_duration / max(0.01, natural_duration)
                segment.render_start = segment.start
                segment.render_end = segment.start + max_duration
        segment.generated_path = str(output_path)
        ratio = segment.duration_factor
        if ratio is not None and ratio < 0.78:
            segment.warning = (
                f"自然语音需压缩到 {ratio:.2f} 倍才能避开下一句；建议精简译文后重生成。"
            )
        return segment

    def segment_cache_key(self, segment: Segment) -> str:
        reference_fingerprint = (
            file_fingerprint(segment.reference_path)
            if segment.reference_path and Path(segment.reference_path).is_file()
            else ""
        )
        emotion_fingerprint = (
            file_fingerprint(segment.emotion_reference_path)
            if segment.emotion_reference_path and Path(segment.emotion_reference_path).is_file()
            else ""
        )
        return stable_hash(
            [
                segment.target_text,
                segment.target_language,
                round(segment.start, 4),
                round(segment.end, 4),
                segment.speaker,
                segment.content_type,
                segment.emotion_text,
                segment.max_render_end,
                self.config.emotion_mode,
                self.config.emotion_strength,
                reference_fingerprint,
                emotion_fingerprint,
                self.config.seed + segment.id,
                "ccd81054de9859faeb19b773fff0e2e1ae9e959e",
            ]
        )

    def unload(self) -> None:
        self._model = None
        release_cuda()
