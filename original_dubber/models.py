from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .platforms import runtime_paths

TARGET_LANGUAGES = {
    "中文": "ZH",
    "English": "EN",
    "日本語": "JA",
    "Español": "ES",
    "العربية": "AR",
}


@dataclass(slots=True)
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Word:
        return cls(**value)


@dataclass(slots=True)
class Segment:
    id: int
    start: float
    end: float
    source_text: str
    target_text: str = ""
    speaker: str = "SPEAKER_00"
    source_language: str = ""
    target_language: str = ""
    enabled: bool = True
    words: list[Word] = field(default_factory=list)
    reference_path: str = ""
    emotion_reference_path: str = ""
    generated_path: str = ""
    natural_duration: float | None = None
    duration_factor: float | None = None
    warning: str = ""
    content_type: str = "speech"
    speech_score: float | None = None
    singing_score: float | None = None
    translation_locked: bool = False
    emotion_text: str = ""
    render_start: float | None = None
    render_end: float | None = None
    max_render_end: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def rendered_duration(self) -> float:
        start = self.start if self.render_start is None else self.render_start
        end = self.end if self.render_end is None else self.render_end
        return max(0.0, end - start)

    @property
    def available_duration(self) -> float:
        end = self.end if self.max_render_end is None else self.max_render_end
        return max(0.0, end - self.start)

    def validate(self) -> None:
        if self.id < 0:
            raise ValueError("segment id must be non-negative")
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError(f"segment {self.id} has a non-finite interval")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid segment interval: {self.start:.3f}-{self.end:.3f}")
        if not self.source_text.strip():
            raise ValueError(f"segment {self.id} has empty source text")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Segment:
        value = dict(value)
        value["words"] = [Word.from_dict(item) for item in value.get("words", [])]
        return cls(**value)


@dataclass(slots=True)
class DubConfig:
    source_video: str
    output_root: str
    project_name: str
    target_language: str = "ZH"
    source_language: str = "auto"
    asr_model: str = "large-v3-turbo"
    asr_hotwords: list[str] = field(default_factory=list)
    asr_replacements: dict[str, str] = field(default_factory=dict)
    asr_runtime_root: str = field(default_factory=lambda: str(runtime_paths().asr))
    model_source: str = "modelscope"
    input_subtitle_path: str = ""
    input_subtitle_mode: str = "none"
    translation_backend: str = "nllb"
    translation_model: str = "facebook/nllb-200-distilled-600M"
    translation_runtime_root: str = field(
        default_factory=lambda: str(runtime_paths().translation)
    )
    translation_context_segments: int = 3
    translation_style: str = "自然、准确、符合人物身份的影视对白"
    translation_glossary: str = ""
    hymt2_quantization: str = "auto"
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    index_api_base: str = "http://127.0.0.1:8000/v1"
    index_api_key: str = ""
    index_model: str = ""
    index_syllables_per_second: float = 4.5
    index_max_tokens: int = 1024
    separate_background: bool = True
    protect_singing_vocals: bool = True
    singing_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    dialogue_mask_margin_ms: int = 100
    diarization: bool = False
    hf_token: str = ""
    emotion_mode: str = "segment"
    emotion_strength: float = 0.72
    enable_qwen_emotion: bool = True
    original_audio_volume: float = 0.18
    background_volume: float = 1.0
    dub_volume: float = 1.0
    seed: int = 42
    use_bf16: bool = True
    use_cuda_kernel: bool = False
    max_reference_seconds: float = 12.0
    max_emotion_reference_seconds: float = 6.0
    min_segment_seconds: float = 0.35
    max_segment_seconds: float = 15.0
    interval_margin_ms: int = 0
    subtitle_mode: str = "soft"
    resume: bool = True
    indextts_dir: str = field(default_factory=lambda: str(runtime_paths().indextts))
    model_dir: str = field(default_factory=lambda: str(runtime_paths().checkpoints))
    config_path: str = field(
        default_factory=lambda: str(runtime_paths().checkpoints / "config.yaml")
    )

    def validate(self) -> None:
        video = Path(self.source_video).expanduser()
        if not video.is_file():
            raise FileNotFoundError(f"找不到视频：{video}")
        if self.target_language not in {"ZH", "EN", "JA", "ES", "AR"}:
            raise ValueError(f"IndexTTS 2.5 不支持目标语言：{self.target_language}")
        if not self.project_name.strip():
            raise ValueError("项目名称不能为空")
        if self.diarization and not self.hf_token:
            raise ValueError("启用说话人分离时必须填写 Hugging Face token")
        if self.translation_backend == "llm" and not (
            self.llm_api_base and self.llm_api_key and self.llm_model
        ):
            raise ValueError("LLM API 翻译需要 API 地址、密钥和模型名")
        if self.translation_backend not in {"nllb", "hymt2", "llm", "none", "index_public", "index", "homura"}:
            raise ValueError(f"未知翻译方式：{self.translation_backend}")
        if self.translation_backend in {"index_public", "index", "homura"}:
            from .index_translation import (
                HOMURA_MODEL,
                PUBLIC_API_BASE,
                PUBLIC_MODEL,
                TRANSLATE_MODEL,
                IndexTranslator,
            )
            public = self.translation_backend == "index_public"
            IndexTranslator(
                api_base=PUBLIC_API_BASE if public else self.index_api_base,
                model=PUBLIC_MODEL if public else (self.index_model or (
                    HOMURA_MODEL if self.translation_backend == "homura" else TRANSLATE_MODEL
                )),
                context_size=self.translation_context_segments,
                glossary=self.translation_glossary,
                syllables_per_second=self.index_syllables_per_second,
                max_tokens=self.index_max_tokens,
            )
        if self.translation_context_segments < 0 or self.translation_context_segments > 12:
            raise ValueError("翻译上下文句数必须在 0～12 之间")
        if self.hymt2_quantization not in {"auto", "4bit", "bf16"}:
            raise ValueError("HY-MT2 精度只支持 auto、4bit 或 bf16")
        if self.emotion_mode not in {"segment", "text", "speaker"}:
            raise ValueError(f"未知情感模式：{self.emotion_mode}")
        if not 0.0 <= self.emotion_strength <= 1.0:
            raise ValueError("情感强度必须在 0～1 之间")
        if self.input_subtitle_mode not in {"none", "source", "translated"}:
            raise ValueError(f"未知字幕导入方式：{self.input_subtitle_mode}")
        if self.input_subtitle_mode != "none":
            subtitle = Path(self.input_subtitle_path).expanduser()
            if not subtitle.is_file():
                raise FileNotFoundError(f"找不到字幕：{subtitle}")
            if subtitle.suffix.lower() not in {".srt", ".vtt", ".ass", ".ssa"}:
                raise ValueError("字幕只支持 SRT、VTT、ASS、SSA")
        if self.input_subtitle_mode == "source" and self.source_language in {"", "auto"}:
            raise ValueError("把上传字幕作为原文时，请明确选择字幕的源语言")
        if self.model_source not in {"modelscope", "huggingface"}:
            raise ValueError("模型来源只支持 modelscope 或 huggingface")
        if not math.isfinite(self.max_segment_seconds) or self.max_segment_seconds <= 0:
            raise ValueError("单段最大时长必须是正数")

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["llm_api_key"] = "***" if self.llm_api_key else ""
        value["index_api_key"] = "***" if self.index_api_key else ""
        value["hf_token"] = "***" if self.hf_token else ""
        return value
