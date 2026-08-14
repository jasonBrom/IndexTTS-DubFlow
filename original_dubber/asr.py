from __future__ import annotations

import json
import math
import queue
import re
import subprocess
import sys
import sysconfig
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Segment, Word
from .platforms import venv_executable
from .utils import release_cuda


@dataclass(slots=True)
class ASRResult:
    segments: list[Segment]
    language: str
    language_probability: float


TERMINAL_PUNCTUATION = "。！？!?．."
CLAUSE_PUNCTUATION = "，、；：,;:"
EXTERNAL_ASR_MODELS = {
    "fireredasr2-aed": "FireRedASR2-AED",
    "fireredasr2s": "FireRedASR2S（VAD/LID/标点完整系统）",
    "qwen3-asr-1.7b": "Qwen3-ASR-1.7B + ForcedAligner",
    "fun-asr-nano-2512": "Fun-ASR-Nano-2512",
}
EXTERNAL_RUNTIME_MARKER = ".ready-v3"
PARENT_ENVIRONMENT_PTH = "indextts_parent_environment.pth"
QWEN_RUNTIME_PROBE = (
    "import transformers; "
    "assert transformers.__version__ == '4.57.6', "
    "f'unexpected transformers {transformers.__version__} at {transformers.__file__}'; "
    "from qwen_asr import Qwen3ASRModel; "
    "print(f'Qwen3-ASR import OK: transformers "
    "{transformers.__version__} ({transformers.__file__})')"
)


def _configure_external_venv_parent_path(
    venv_root: Path,
    python: Path,
    parent_site: str | Path,
) -> Path:
    """Share parent-only packages without letting them shadow ASR dependencies."""
    legacy_bridge = venv_root / PARENT_ENVIRONMENT_PTH
    legacy_bridge.unlink(missing_ok=True)

    config = venv_root / "pyvenv.cfg"
    if config.is_file():
        source = config.read_text(encoding="utf-8")
        updated, replacements = re.subn(
            r"^include-system-site-packages\s*=\s*true\s*$",
            "include-system-site-packages = false",
            source,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not replacements and not re.search(
            r"^include-system-site-packages\s*=", source, re.IGNORECASE | re.MULTILINE
        ):
            updated = source.rstrip() + "\ninclude-system-site-packages = false\n"
        if updated != source:
            config.write_text(updated, encoding="utf-8")

    external_site = subprocess.check_output(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        text=True,
    ).strip()
    bridge = Path(external_site) / PARENT_ENVIRONMENT_PTH
    bridge.write_text(str(Path(parent_site).resolve()) + "\n", encoding="utf-8")
    return bridge


def parse_hotword_spec(*values: str) -> tuple[list[str], dict[str, str]]:
    terms: list[str] = []
    replacements: dict[str, str] = {}
    for value in values:
        for raw_line in re.split(r"[\n,，]+", value or ""):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" in line:
                wrong, correct = (part.strip() for part in line.split("=>", 1))
                if wrong and correct:
                    replacements[wrong] = correct
                continue
            if line not in terms:
                terms.append(line)
    return terms, replacements


def apply_hotword_replacements(
    segments: list[Segment], replacements: dict[str, str]
) -> list[Segment]:
    if not replacements:
        return segments
    for segment in segments:
        for wrong, correct in replacements.items():
            segment.source_text = segment.source_text.replace(wrong, correct)
            for word in segment.words:
                word.text = word.text.replace(wrong, correct)
    return segments


class FasterWhisperASR:
    def __init__(
        self, model_name: str = "large-v3-turbo", device: str = "auto"
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "缺少 faster-whisper，请先运行项目安装脚本。"
            ) from exc
        import torch

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(
            self.model_name, device=device, compute_type=compute_type
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        source_language: str = "auto",
        min_segment_seconds: float = 0.35,
        hotwords: list[str] | None = None,
        replacements: dict[str, str] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> ASRResult:
        if progress:
            progress(0.0, f"下载/加载 Whisper {self.model_name}（首次运行较慢）")
        self._load()
        if progress:
            progress(0.02, f"Whisper {self.model_name} 已加载，开始识别")
        language = None if source_language in {"", "auto"} else source_language.lower()
        transcribe_kwargs: dict[str, Any] = {
            "language": language,
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 300},
            "word_timestamps": True,
            "condition_on_previous_text": True,
        }
        native_hotwords = " ".join(hotwords or [])
        if native_hotwords:
            import inspect

            parameters = inspect.signature(self.model.transcribe).parameters
            if "hotwords" in parameters:
                transcribe_kwargs["hotwords"] = native_hotwords
            else:
                transcribe_kwargs["initial_prompt"] = f"关键词：{native_hotwords}"
        iterable, info = self.model.transcribe(
            str(audio_path),
            **transcribe_kwargs,
        )
        values: list[Segment] = []
        for index, item in enumerate(iterable):
            start = float(item.start)
            end = float(item.end)
            text = item.text.strip()
            if not text or end - start < min_segment_seconds:
                continue
            words = [
                Word(
                    start=float(word.start if word.start is not None else start),
                    end=float(word.end if word.end is not None else end),
                    text=word.word,
                    probability=float(word.probability)
                    if word.probability is not None
                    else None,
                )
                for word in (item.words or [])
            ]
            values.append(
                Segment(
                    id=len(values),
                    start=start,
                    end=end,
                    source_text=text,
                    source_language=info.language,
                    words=words,
                )
            )
            if progress:
                progress(min(0.95, 0.02 + index * 0.01), f"识别片段 {len(values)}")
        return ASRResult(
            segments=apply_hotword_replacements(values, replacements or {}),
            language=info.language,
            language_probability=float(info.language_probability),
        )

    def unload(self) -> None:
        self.model = None
        release_cuda()


def _run_visible(
    command: list[str],
    *,
    label: str,
    progress: Callable[[float, str], None] | None,
    cwd: Path | None = None,
) -> None:
    print(f"[ASR] ▶ {label}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    lines: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            lines.put(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started = time.monotonic()
    last_message = label
    next_heartbeat = started
    while process.poll() is None or reader.is_alive() or not lines.empty():
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            line = ""
        if line:
            print(line, end="", flush=True)
            last_message = line.strip()[-160:] or last_message
        elapsed = time.monotonic() - started
        if progress and time.monotonic() >= next_heartbeat:
            progress(0.0, f"{label}：{last_message}（{int(elapsed)} 秒）")
            next_heartbeat = time.monotonic() + 10
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{label}失败（退出码 {return_code}），请查看 Web 后台日志")
    print(f"[ASR] ✅ {label}，耗时 {time.monotonic() - started:.1f} 秒", flush=True)


class ExternalASR:
    def __init__(
        self,
        backend: str,
        *,
        runtime_root: str | Path,
        model_source: str,
    ) -> None:
        if backend not in EXTERNAL_ASR_MODELS:
            raise ValueError(f"未知外部 ASR：{backend}")
        self.backend = backend
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.model_source = model_source

    @property
    def backend_root(self) -> Path:
        return self.runtime_root / self.backend

    @property
    def python(self) -> Path:
        return venv_executable(self.backend_root / ".venv")

    def _pip(
        self, packages: list[str], progress: Callable[[float, str], None] | None
    ) -> None:
        command = [str(self.python), "-m", "pip", "install", *packages]
        _run_visible(
            command,
            label=f"安装 {EXTERNAL_ASR_MODELS[self.backend]} 运行依赖",
            progress=progress,
        )

    def _ensure_runtime(self, progress: Callable[[float, str], None] | None) -> None:
        root = self.backend_root
        marker = root / EXTERNAL_RUNTIME_MARKER
        venv_root = root / ".venv"
        root.mkdir(parents=True, exist_ok=True)
        if not self.python.is_file():
            _run_visible(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    str(venv_root),
                ],
                label=f"创建隔离环境 {self.backend}",
                progress=progress,
            )
        _configure_external_venv_parent_path(
            venv_root,
            self.python,
            sysconfig.get_paths()["purelib"],
        )
        if marker.is_file():
            return
        if self.backend == "qwen3-asr-1.7b":
            self._pip(["qwen-asr==0.0.6", "modelscope>=1.28,<2"], progress)
        elif self.backend == "fun-asr-nano-2512":
            self._pip(
                [
                    "funasr==1.3.26",
                    "modelscope>=1.28,<2",
                    "tiktoken",
                    "huggingface_hub",
                ],
                progress,
            )
        else:
            repository = root / "FireRedASR2S"
            if not (repository / ".git").is_dir():
                _run_visible(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/FireRedTeam/FireRedASR2S.git",
                        str(repository),
                    ],
                    label="获取 FireRedASR2S 官方代码",
                    progress=progress,
                )
            self._pip(
                [
                    "transformers==4.51.3",
                    "numpy==1.26.1",
                    "cn2an==0.5.23",
                    "kaldiio==2.18.0",
                    "kaldi_native_fbank==1.15",
                    "sentencepiece==0.1.99",
                    "soundfile==0.12.1",
                    "textgrid",
                    "peft>=0.13.2",
                    "modelscope>=1.28,<2",
                    "huggingface_hub",
                ],
                progress,
            )
            self._pip(["--no-deps", "-e", str(repository)], progress)
        if self.backend == "qwen3-asr-1.7b":
            _run_visible(
                [
                    str(self.python),
                    "-c",
                    QWEN_RUNTIME_PROBE,
                ],
                label="验证 Qwen3-ASR 隔离环境",
                progress=progress,
            )
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        source_language: str = "auto",
        min_segment_seconds: float = 0.35,
        hotwords: list[str] | None = None,
        replacements: dict[str, str] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> ASRResult:
        self._ensure_runtime(progress)
        output = self.backend_root / "last_result.json"
        project_root = Path(__file__).resolve().parents[1]
        command = [
            str(self.python),
            "-u",
            str(project_root / "scripts" / "asr_worker.py"),
            "--backend",
            self.backend,
            "--audio",
            str(Path(audio_path).resolve()),
            "--output",
            str(output),
            "--runtime-root",
            str(self.backend_root),
            "--model-source",
            self.model_source,
            "--source-language",
            source_language,
            "--hotwords-json",
            json.dumps(hotwords or [], ensure_ascii=False),
        ]
        _run_visible(
            command,
            label=f"运行 {EXTERNAL_ASR_MODELS[self.backend]}（首次会下载模型）",
            progress=progress,
        )
        value = json.loads(output.read_text(encoding="utf-8"))
        segments = [
            segment
            for item in value.get("segments", [])
            if (segment := Segment.from_dict(item)).duration >= min_segment_seconds
        ]
        apply_hotword_replacements(segments, replacements or {})
        return ASRResult(
            segments=segments,
            language=value.get("language") or source_language,
            language_probability=float(value.get("language_probability", 1.0)),
        )

    def unload(self) -> None:
        release_cuda()


def create_asr(
    model_name: str,
    *,
    runtime_root: str | Path,
    model_source: str,
) -> FasterWhisperASR | ExternalASR:
    if model_name in EXTERNAL_ASR_MODELS:
        return ExternalASR(
            model_name, runtime_root=runtime_root, model_source=model_source
        )
    whisper_model = model_name.removeprefix("whisper:")
    return FasterWhisperASR(whisper_model)


def assign_speakers_pyannote(
    audio_path: str | Path,
    segments: list[Segment],
    *,
    hf_token: str,
    model_id: str = "pyannote/speaker-diarization-3.1",
) -> list[Segment]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "说话人分离未安装。请使用 --with-diarization 重新运行安装脚本，"
            "或在 Colab 中开启 INSTALL_DIARIZATION。"
        ) from exc
    import torch

    try:
        pipeline = Pipeline.from_pretrained(model_id, token=hf_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    diarization_output = pipeline(str(audio_path))
    diarization = getattr(diarization_output, "speaker_diarization", diarization_output)
    turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    for segment in segments:
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for start, end, speaker in turns:
            overlap = max(0.0, min(segment.end, end) - max(segment.start, start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        segment.speaker = best_speaker
    del pipeline
    release_cuda()
    return segments


def split_long_segments(segments: list[Segment], max_seconds: float) -> list[Segment]:
    """Split on sentence/clause/silence boundaries before falling back to a word boundary."""
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    result: list[Segment] = []

    def append_words(source: Segment, words: list[Word]) -> None:
        text = "".join(word.text for word in words).strip()
        if not text:
            text = source.source_text
        result.append(
            Segment(
                id=len(result),
                start=max(source.start, words[0].start),
                end=min(source.end, words[-1].end),
                source_text=text,
                source_language=source.source_language,
                words=words,
                speaker=source.speaker,
                content_type=source.content_type,
            )
        )

    def sentence_groups(words: list[Word]) -> list[list[Word]]:
        groups: list[list[Word]] = []
        current: list[Word] = []
        for word in words:
            current.append(word)
            if word.text.rstrip().endswith(tuple(TERMINAL_PUNCTUATION)):
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups

    def split_group(words: list[Word]) -> list[list[Word]]:
        pieces: list[list[Word]] = []
        expanded: list[Word] = []
        for word in words:
            if word.end - word.start <= max_seconds:
                expanded.append(word)
                continue
            count = math.ceil((word.end - word.start) / max_seconds)
            for index in range(count):
                start = word.start + (word.end - word.start) * index / count
                end = word.start + (word.end - word.start) * (index + 1) / count
                text_start = round(len(word.text) * index / count)
                text_end = round(len(word.text) * (index + 1) / count)
                expanded.append(
                    Word(start=start, end=end, text=word.text[text_start:text_end])
                )
        remaining = expanded
        while remaining and remaining[-1].end - remaining[0].start > max_seconds:
            candidates: list[tuple[int, float, int]] = []
            for index, word in enumerate(remaining[:-1], start=1):
                elapsed = word.end - remaining[0].start
                if elapsed > max_seconds:
                    break
                gap = max(0.0, remaining[index].start - word.end)
                clause = word.text.rstrip().endswith(tuple(CLAUSE_PUNCTUATION))
                priority = 2 if clause else (1 if gap >= 0.28 else 0)
                if elapsed >= min(2.0, max_seconds * 0.35):
                    candidates.append((priority, elapsed, index))
            if candidates:
                best_priority = max(item[0] for item in candidates)
                pool = [item for item in candidates if item[0] == best_priority]
                _, _, cut = min(pool, key=lambda item: abs(max_seconds - item[1]))
            else:
                cut = 1
                while (
                    cut < len(remaining)
                    and remaining[cut].end - remaining[0].start <= max_seconds
                ):
                    cut += 1
                cut = max(1, cut - 1)
            pieces.append(remaining[:cut])
            remaining = remaining[cut:]
        if remaining:
            pieces.append(remaining)
        return pieces

    def estimated_words(source: Segment) -> list[Word]:
        """Build a text-aligned fallback when an ASR only exposes sentence timestamps."""
        text = source.source_text.strip()
        if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text):
            tokens = [character for character in text if not character.isspace()]
        else:
            tokens = re.findall(r"\S+\s*", text)
        if not tokens:
            tokens = [text]
        weights = [max(1, len(token.strip())) for token in tokens]
        total_weight = sum(weights)
        cursor = source.start
        words: list[Word] = []
        for index, (token, weight) in enumerate(zip(tokens, weights, strict=False)):
            end = (
                source.end
                if index == len(tokens) - 1
                else cursor + source.duration * weight / total_weight
            )
            words.append(Word(start=cursor, end=end, text=token))
            cursor = end
        return words

    for segment in segments:
        if segment.words:
            groups = sentence_groups(segment.words)
            if len(groups) == 1 and segment.duration <= max_seconds:
                segment.id = len(result)
                result.append(segment)
                continue
            for group in groups:
                for piece in split_group(group):
                    append_words(segment, piece)
            continue

        if segment.duration <= max_seconds:
            segment.id = len(result)
            result.append(segment)
            continue

        fallback_words = estimated_words(segment)
        for group in sentence_groups(fallback_words):
            for piece in split_group(group):
                append_words(segment, piece)
    return result


def merge_incomplete_segments(
    segments: list[Segment],
    *,
    max_seconds: float,
    max_gap_seconds: float = 0.45,
) -> list[Segment]:
    """Merge adjacent ASR fragments when the first fragment does not finish a sentence."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    merged: list[Segment] = []
    for incoming in ordered:
        if not merged:
            merged.append(incoming)
            continue
        previous = merged[-1]
        gap = max(0.0, incoming.start - previous.end)
        finishes_sentence = previous.source_text.rstrip().endswith(
            tuple(TERMINAL_PUNCTUATION)
        )
        can_merge = (
            not finishes_sentence
            and gap <= max_gap_seconds
            and incoming.end - previous.start <= max_seconds
            and previous.content_type == incoming.content_type
            and previous.speaker == incoming.speaker
        )
        if not can_merge:
            merged.append(incoming)
            continue
        previous.end = max(previous.end, incoming.end)
        if previous.source_text and incoming.source_text:
            separator = (
                ""
                if (
                    previous.source_text[-1:].isascii() is False
                    or incoming.source_text[:1].isascii() is False
                    or incoming.source_text.startswith(" ")
                )
                else " "
            )
            previous.source_text = (
                previous.source_text.rstrip()
                + separator
                + incoming.source_text.lstrip()
            )
        previous.words.extend(incoming.words)
        if previous.words:
            previous.start = min(previous.start, previous.words[0].start)
            previous.end = max(previous.end, previous.words[-1].end)
    for index, segment in enumerate(merged):
        segment.id = index
    return merged


def normalize_asr_segments(
    segments: list[Segment], max_seconds: float
) -> list[Segment]:
    return split_long_segments(
        merge_incomplete_segments(segments, max_seconds=max_seconds),
        max_seconds,
    )
