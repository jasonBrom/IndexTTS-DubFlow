from __future__ import annotations

import argparse
import json
import os
import re
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from original_dubber.asr import normalize_asr_segments
from original_dubber.models import Segment, Word
from original_dubber.utils import atomic_write_json

LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "es": "Spanish",
    "ar": "Arabic",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "pt": "Portuguese",
    "vi": "Vietnamese",
}
LANGUAGE_CODES = {value.lower(): key for key, value in LANGUAGE_NAMES.items()}
LANGUAGE_CODES.update({"中文": "zh", "英语": "en", "日语": "ja", "粤语": "yue"})


def detect_language(text: str, fallback: str = "zh") -> str:
    if re.search(r"[ぁ-んァ-ン]", text):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u0600-\u06ff]", text):
        return "ar"
    normalized = LANGUAGE_CODES.get(str(fallback).lower(), str(fallback).lower())
    return normalized if normalized not in {"", "auto", "none"} else "en"


def model_path(
    runtime_root: Path,
    *,
    folder: str,
    huggingface_id: str,
    modelscope_id: str,
    source: str,
    direct: bool = False,
) -> Path:
    destination = runtime_root / folder if direct else runtime_root / "models" / folder
    complete = destination / ".download_complete"
    if complete.is_file():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[ASR worker] 下载模型 {folder} → {destination}", flush=True)
    if source == "modelscope":
        from modelscope import snapshot_download

        snapshot_download(modelscope_id, local_dir=str(destination))
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(huggingface_id, local_dir=str(destination))
    complete.write_text("ok\n", encoding="utf-8")
    return destination


def silence_chunks(
    audio_path: Path, output_dir: Path, max_seconds: float
) -> list[tuple[Path, float]]:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1)
    total_seconds = len(mono) / sample_rate
    if total_seconds <= max_seconds:
        return [(audio_path, 0.0)]
    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries = [0]
    cursor = 0
    target_samples = int(max_seconds * sample_rate)
    search_samples = int(min(12.0, max_seconds * 0.2) * sample_rate)
    window = max(1, int(0.20 * sample_rate))
    while cursor + target_samples < len(mono):
        target = cursor + target_samples
        low = max(
            cursor + int(max_seconds * 0.65 * sample_rate), target - search_samples
        )
        high = min(len(mono) - window, target + search_samples)
        best = target
        best_energy = float("inf")
        step = max(1, window // 2)
        for candidate in range(low, high + 1, step):
            energy = float(np.mean(np.abs(mono[candidate : candidate + window])))
            if energy < best_energy:
                best_energy = energy
                best = candidate + window // 2
        if best <= cursor:
            best = target
        boundaries.append(best)
        cursor = best
    boundaries.append(len(mono))
    chunks: list[tuple[Path, float]] = []
    for index, (start, end) in enumerate(pairwise(boundaries)):
        output = output_dir / f"chunk_{index:04d}.wav"
        sf.write(output, audio[start:end], sample_rate, subtype="PCM_16")
        chunks.append((output, start / sample_rate))
    return chunks


def timestamp_segments(
    items: list[Any],
    *,
    offset: float,
    language: str,
    fallback_text: str = "",
) -> list[Segment]:
    words: list[Word] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("word") or "")
            start = item.get("start_time", item.get("start", item.get("start_ms", 0)))
            end = item.get("end_time", item.get("end", item.get("end_ms", start)))
            if "start_ms" in item or "end_ms" in item:
                start, end = float(start) / 1000, float(end) / 1000
        elif hasattr(item, "text"):
            text = str(item.text)
            start = float(getattr(item, "start_time", getattr(item, "start", 0)))
            end = float(getattr(item, "end_time", getattr(item, "end", start)))
        elif len(item) >= 3 and isinstance(item[0], str):
            text, start, end = str(item[0]), float(item[1]), float(item[2])
        else:
            continue
        if float(end) > float(start):
            words.append(
                Word(start=offset + float(start), end=offset + float(end), text=text)
            )
    if not words:
        return []
    segment = Segment(
        id=0,
        start=words[0].start,
        end=words[-1].end,
        source_text="".join(word.text for word in words).strip()
        or fallback_text.strip(),
        source_language=language,
        words=words,
    )
    return normalize_asr_segments([segment], max_seconds=15.0)


def text_tokens(text: str, count: int) -> list[str]:
    compact = [char for char in text if not char.isspace()]
    words = re.findall(r"\S+\s*", text)
    if len(compact) == count:
        return compact
    if len(words) == count:
        return words
    if count <= 1:
        return [text]
    tokens: list[str] = []
    for index in range(count):
        start = round(len(text) * index / count)
        end = round(len(text) * (index + 1) / count)
        tokens.append(text[start:end])
    return tokens


def pair_plain_timestamps(
    text: str, timestamps: list[Any]
) -> list[tuple[str, float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in timestamps:
        if isinstance(item, dict):
            start = item.get("start", item.get("start_ms", 0))
            end = item.get("end", item.get("end_ms", start))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = float(item[-2]), float(item[-1])
        else:
            continue
        pairs.append((float(start) / 1000, float(end) / 1000))
    tokens = text_tokens(text, len(pairs))
    return [
        (token, start, end)
        for token, (start, end) in zip(tokens, pairs, strict=False)
    ]


def run_qwen(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from qwen_asr import Qwen3ASRModel

    asr_path = model_path(
        args.runtime_root,
        folder="Qwen3-ASR-1.7B",
        huggingface_id="Qwen/Qwen3-ASR-1.7B",
        modelscope_id="Qwen/Qwen3-ASR-1.7B",
        source=args.model_source,
    )
    aligner_path = model_path(
        args.runtime_root,
        folder="Qwen3-ForcedAligner-0.6B",
        huggingface_id="Qwen/Qwen3-ForcedAligner-0.6B",
        modelscope_id="Qwen/Qwen3-ForcedAligner-0.6B",
        source=args.model_source,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device != "cpu" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device != "cpu" else torch.float32)
    )
    model = Qwen3ASRModel.from_pretrained(
        str(asr_path),
        dtype=dtype,
        device_map=device,
        forced_aligner=str(aligner_path),
        forced_aligner_kwargs={"dtype": dtype, "device_map": device},
        max_inference_batch_size=1,
        max_new_tokens=4096,
    )
    requested_language = (
        None
        if args.source_language in {"", "auto"}
        else LANGUAGE_NAMES.get(args.source_language, args.source_language)
    )
    context = "、".join(args.hotwords)
    all_segments: list[Segment] = []
    detected = args.source_language
    chunks = silence_chunks(args.audio, args.runtime_root / "chunks_qwen", 240.0)
    for index, (chunk, offset) in enumerate(chunks, start=1):
        print(
            f"[ASR worker] Qwen3-ASR 分块 {index}/{len(chunks)}，偏移 {offset:.1f}s",
            flush=True,
        )
        result = model.transcribe(
            audio=str(chunk),
            language=requested_language,
            context=context,
            return_time_stamps=True,
        )[0]
        detected = LANGUAGE_CODES.get(
            str(result.language or detected).lower(),
            str(result.language or detected).lower(),
        )
        all_segments.extend(
            timestamp_segments(
                list(result.time_stamps or []),
                offset=offset,
                language=detect_language(result.text, detected),
                fallback_text=result.text,
            )
        )
    return {
        "segments": [item.to_dict() for item in all_segments],
        "language": detected,
        "language_probability": 1.0,
    }


def prepare_firered_models(args: argparse.Namespace, system: bool) -> Path:
    repository = args.runtime_root / "FireRedASR2S"
    required = [
        ("FireRedASR2-AED", "FireRedTeam/FireRedASR2-AED", "xukaituo/FireRedASR2-AED"),
    ]
    if system:
        required += [
            ("FireRedVAD", "FireRedTeam/FireRedVAD", "xukaituo/FireRedVAD"),
            ("FireRedLID", "FireRedTeam/FireRedLID", "xukaituo/FireRedLID"),
            ("FireRedPunc", "FireRedTeam/FireRedPunc", "xukaituo/FireRedPunc"),
        ]
    for folder, hf_id, ms_id in required:
        model_path(
            repository / "pretrained_models",
            folder=folder,
            huggingface_id=hf_id,
            modelscope_id=ms_id,
            source=args.model_source,
            direct=True,
        )
    return repository


def run_firered_aed(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    repository = prepare_firered_models(args, system=False)
    sys.path.insert(0, str(repository))
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    config = FireRedAsr2Config(
        use_gpu=torch.cuda.is_available(),
        use_half=False,
        beam_size=3,
        nbest=1,
        decode_max_len=0,
        softmax_smoothing=1.25,
        aed_length_penalty=0.6,
        eos_penalty=1.0,
        return_timestamp=True,
    )
    model = FireRedAsr2.from_pretrained(
        "aed", str(repository / "pretrained_models" / "FireRedASR2-AED"), config
    )
    chunks = silence_chunks(args.audio, args.runtime_root / "chunks_firered", 50.0)
    all_segments: list[Segment] = []
    texts: list[str] = []
    for index, (chunk, offset) in enumerate(chunks, start=1):
        print(f"[ASR worker] FireRedASR2-AED 分块 {index}/{len(chunks)}", flush=True)
        result = model.transcribe([f"chunk_{index}"], [str(chunk)])[0]
        text = str(result.get("text", ""))
        texts.append(text)
        all_segments.extend(
            timestamp_segments(
                list(result.get("timestamp") or []),
                offset=offset,
                language=detect_language(text, args.source_language),
                fallback_text=text,
            )
        )
    language = detect_language("".join(texts), args.source_language)
    return {
        "segments": [item.to_dict() for item in all_segments],
        "language": language,
        "language_probability": 1.0,
    }


def run_firered_system(args: argparse.Namespace) -> dict[str, Any]:
    repository = prepare_firered_models(args, system=True)
    sys.path.insert(0, str(repository))
    os.chdir(repository)
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig

    system = FireRedAsr2System(FireRedAsr2SystemConfig())
    result = system.process(str(args.audio))
    raw_words = result.get("words") or []
    words = [
        Word(
            start=float(item["start_ms"]) / 1000,
            end=float(item["end_ms"]) / 1000,
            text=str(item["text"]),
        )
        for item in raw_words
    ]
    segments: list[Segment] = []
    detected = args.source_language
    confidence = 1.0
    for sentence in result.get("sentences") or []:
        start = float(sentence["start_ms"]) / 1000
        end = float(sentence["end_ms"]) / 1000
        text = str(sentence.get("text") or "").strip()
        sentence_words = [
            word for word in words if word.end > start and word.start < end
        ]
        language_label = str(sentence.get("lang") or detected).split()[0]
        detected = language_label
        confidence = float(sentence.get("lang_confidence", confidence))
        if text and end > start:
            segments.append(
                Segment(
                    id=len(segments),
                    start=start,
                    end=end,
                    source_text=text,
                    source_language=language_label,
                    words=sentence_words,
                )
            )
    return {
        "segments": [item.to_dict() for item in segments],
        "language": detected,
        "language_probability": confidence,
    }


def run_funasr(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from funasr import AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    hub = "ms" if args.model_source == "modelscope" else "hf"
    model = AutoModel(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        hub=hub,
        trust_remote_code=True,
        disable_update=True,
    )
    language = (
        "auto"
        if args.source_language in {"", "auto"}
        else LANGUAGE_NAMES.get(args.source_language, args.source_language)
    )
    result = model.generate(
        input=str(args.audio),
        cache={},
        batch_size=1,
        language=language,
        hotwords=args.hotwords,
    )[0]
    detected = LANGUAGE_CODES.get(
        str(result.get("language") or args.source_language).lower(),
        str(result.get("language") or args.source_language).lower(),
    )
    segments: list[Segment] = []
    for sentence in result.get("sentence_info") or []:
        start_ms = float(sentence.get("start", sentence.get("start_ms", 0)))
        end_ms = float(sentence.get("end", sentence.get("end_ms", start_ms)))
        start, end = start_ms / 1000, end_ms / 1000
        text = str(sentence.get("text") or "").strip()
        if text and end > start:
            raw_sentence_timestamps = (
                sentence.get("timestamps") or sentence.get("timestamp") or []
            )
            triples = pair_plain_timestamps(text, list(raw_sentence_timestamps))
            if triples and triples[0][1] < start - 0.05:
                triples = [
                    (token, word_start + start, word_end + start)
                    for token, word_start, word_end in triples
                ]
            words = [
                Word(start=word_start, end=word_end, text=token)
                for token, word_start, word_end in triples
                if word_end > word_start and word_end <= end + 0.25
            ]
            segments.append(
                Segment(
                    id=len(segments),
                    start=start,
                    end=end,
                    source_text=text,
                    source_language=detect_language(text, detected),
                    words=words,
                )
            )
    if not segments:
        text = str(result.get("text") or "").strip()
        raw_timestamps = result.get("timestamps") or result.get("timestamp") or []
        triples = pair_plain_timestamps(text, list(raw_timestamps))
        segments = timestamp_segments(
            triples,
            offset=0.0,
            language=detect_language(text, detected),
            fallback_text=text,
        )
    output_language = (
        segments[0].source_language if segments else detect_language(text, detected)
    )
    return {
        "segments": [item.to_dict() for item in segments],
        "language": output_language,
        "language_probability": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--model-source", choices=["modelscope", "huggingface"], required=True
    )
    parser.add_argument("--source-language", default="auto")
    parser.add_argument("--hotwords-json", default="[]")
    args = parser.parse_args()
    args.hotwords = json.loads(args.hotwords_json)
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    print(f"[ASR worker] backend={args.backend} source={args.model_source}", flush=True)
    if args.backend == "qwen3-asr-1.7b":
        value = run_qwen(args)
    elif args.backend == "fun-asr-nano-2512":
        value = run_funasr(args)
    elif args.backend == "fireredasr2-aed":
        value = run_firered_aed(args)
    elif args.backend == "fireredasr2s":
        value = run_firered_system(args)
    else:
        raise ValueError(f"未知后端：{args.backend}")
    atomic_write_json(args.output, value)
    print(
        f"[ASR worker] ✅ 输出 {len(value.get('segments', []))} 段 → {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
