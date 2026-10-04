from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import Segment
from .platforms import runtime_paths, venv_executable
from .utils import release_cuda

NLLB_LANGUAGE_CODES = {
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "yue": "yue_Hant",
    "zh": "zho_Hans",
}

TARGET_NLLB_CODES = {
    "ZH": "zho_Hans",
    "EN": "eng_Latn",
    "JA": "jpn_Jpan",
    "ES": "spa_Latn",
    "AR": "arb_Arab",
}

TARGET_LANGUAGE_NAMES = {
    "ZH": "Simplified Chinese",
    "EN": "English",
    "JA": "Japanese",
    "ES": "Spanish",
    "AR": "Arabic",
}

SOURCE_LANGUAGE_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese",
    "ru": "Russian", "vi": "Vietnamese", "yue": "Cantonese", "zh": "Chinese",
}


class NLLBTranslator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

    def translate(
        self,
        segments: list[Segment],
        *,
        source_language: str,
        target_language: str,
        context_segments: list[Segment] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[Segment]:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("缺少 transformers，IndexTTS 环境可能没有正确安装。") from exc
        import torch

        source_code = NLLB_LANGUAGE_CODES.get(source_language.lower())
        target_code = TARGET_NLLB_CODES[target_language]
        if source_code is None:
            raise ValueError(
                f"NLLB 默认映射里没有源语言 {source_language!r}。可改用 LLM API，或在时间轴手工填写译文。"
            )
        if progress:
            progress(0.0, f"下载/加载 NLLB 翻译模型 {self.model_name}（首次运行较慢）")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang=source_code)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        actual_device = self.device
        if actual_device == "auto":
            actual_device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(actual_device).eval()
        if progress:
            progress(0.02, f"NLLB 已加载到 {actual_device}，开始翻译")
        texts = [segment.source_text for segment in segments]
        batch_size = 8 if actual_device == "cuda" else 4
        completed = 0
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(actual_device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
                    max_new_tokens=256,
                    num_beams=4,
                )
            translated = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for segment, text in zip(
                segments[offset : offset + batch_size], translated, strict=False
            ):
                segment.target_text = text.strip()
                segment.target_language = target_language
            completed += len(batch)
            if progress:
                progress(completed / max(1, len(texts)), f"翻译 {completed}/{len(texts)}")
        del model, tokenizer
        release_cuda()
        return segments


@dataclass(slots=True)
class ChatCompletionsTranslator:
    api_base: str
    api_key: str
    model: str
    timeout: int = 120

    def _endpoint(self) -> str:
        base = self.api_base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def translate(
        self,
        segments: list[Segment],
        *,
        source_language: str,
        target_language: str,
        context_segments: list[Segment] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[Segment]:
        import requests

        target_name = TARGET_LANGUAGE_NAMES[target_language]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        full_timeline = context_segments or segments
        for index, segment in enumerate(segments):
            timeline_index = next(
                (offset for offset, item in enumerate(full_timeline) if item is segment), index
            )
            previous = full_timeline[max(0, timeline_index - 3) : timeline_index]
            following = full_timeline[timeline_index + 1 : timeline_index + 3]
            context = "\n".join(
                [f"Previous {item.speaker}: {item.source_text}" for item in previous]
                + [f"Next {item.speaker}: {item.source_text}" for item in following]
            )
            prompt = (
                f"Translate the dialogue into {target_name}. The spoken result must fit naturally in "
                f"approximately {segment.duration:.2f} seconds. Preserve meaning, tone, names and numbers, "
                "but use concise natural wording when needed. Return only the translated line, without quotes.\n\n"
                f"Source language: {source_language}\nSpeaker: {segment.speaker}\n"
                f"Context:\n{context or '(none)'}\nSource: {segment.source_text}"
            )
            payload = {
                "model": self.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You are a professional audiovisual dialogue translator."},
                    {"role": "user", "content": prompt},
                ],
            }
            response = requests.post(
                self._endpoint(), headers=headers, data=json.dumps(payload), timeout=self.timeout
            )
            if response.status_code >= 400:
                raise RuntimeError(f"翻译 API 返回 {response.status_code}：{response.text[:500]}")
            value = response.json()
            segment.target_text = value["choices"][0]["message"]["content"].strip().strip('"')
            segment.target_language = target_language
            if progress:
                progress((index + 1) / max(1, len(segments)), f"翻译 {index + 1}/{len(segments)}")
        return segments


def _run_visible(
    command: list[str],
    *,
    label: str,
    progress: Callable[[float, str], None] | None,
    env: dict[str, str] | None = None,
) -> None:
    print(f"[Translation] ▶ {label}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    lines: queue.Queue[str] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            lines.put(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started = time.monotonic()
    next_heartbeat = started
    last_line = label
    tail: deque[str] = deque(maxlen=30)
    while process.poll() is None or reader.is_alive() or not lines.empty():
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            line = ""
        if line:
            print(line, end="", flush=True)
            last_line = line.strip()[-180:] or last_line
            tail.append(line.rstrip())
        if progress and time.monotonic() >= next_heartbeat:
            progress(0.0, f"{label}：{last_line}（{int(time.monotonic() - started)} 秒）")
            next_heartbeat = time.monotonic() + 10
    if process.wait():
        details = "\n".join(tail)
        if "No space left on device" in details or "os error 28" in details:
            raise RuntimeError(
                f"{label}失败：磁盘空间不足。请运行 Notebook 的“磁盘诊断与安全清理”单元，"
                "不要删除 Qwen ASR 模型或 HY-MT2 已完成的分片；清理后可直接重试。\n"
                f"最近日志：\n{details}"
            )
        raise RuntimeError(f"{label}失败。最近日志：\n{details}")
    print(f"[Translation] ✅ {label}，耗时 {time.monotonic() - started:.1f} 秒", flush=True)


@dataclass(slots=True)
class HyMT2Translator:
    model_name: str = "tencent/Hy-MT2-7B"
    runtime_root: str = field(default_factory=lambda: str(runtime_paths().translation))
    quantization: str = "auto"
    context_size: int = 3
    style: str = "自然、准确、符合人物身份的影视对白"
    glossary: str = ""

    @property
    def root(self) -> Path:
        return Path(self.runtime_root).expanduser().resolve() / "hymt2"

    @property
    def python(self) -> Path:
        return venv_executable(self.root / ".venv")

    @property
    def hf_home(self) -> Path:
        configured = os.environ.get("HYMT_HF_HOME") or os.environ.get("HF_HOME")
        return Path(configured).expanduser().resolve() if configured else self.root / "hf_cache"

    def _worker_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["HF_HOME"] = str(self.hf_home)
        env["HF_HUB_CACHE"] = str(self.hf_home / "hub")
        env["HF_HUB_DISABLE_XET"] = env.get("HF_HUB_DISABLE_XET", "1")
        env["PIP_NO_CACHE_DIR"] = "1"
        return env

    def _ensure_runtime(self, progress: Callable[[float, str], None] | None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.hf_home.mkdir(parents=True, exist_ok=True)
        worker_env = self._worker_env()
        marker = self.root / ".ready-v2"
        if not self.python.is_file():
            _run_visible(
                [sys.executable, "-m", "venv", "--system-site-packages", str(self.root / ".venv")],
                label="创建 HY-MT2 隔离环境",
                progress=progress,
                env=worker_env,
            )
        if marker.is_file():
            return
        _run_visible(
            [
                str(self.python), "-m", "pip", "install",
                "--no-cache-dir",
                "transformers>=5.6.0", "accelerate>=1.10", "bitsandbytes>=0.46",
                "sentencepiece", "huggingface_hub>=0.34",
            ],
            label="安装 HY-MT2-7B 推理依赖",
            progress=progress,
            env=worker_env,
        )
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"), encoding="utf-8")

    def translate(
        self,
        segments: list[Segment],
        *,
        source_language: str,
        target_language: str,
        context_segments: list[Segment] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[Segment]:
        self._ensure_runtime(progress)
        full = context_segments or segments
        pending_ids = {id(item) for item in segments}
        payload_segments = []
        for item in full:
            payload_segments.append(
                {
                    "id": item.id,
                    "speaker": item.speaker,
                    "start": item.start,
                    "end": item.end,
                    "duration": item.duration,
                    "source_text": item.source_text,
                    "target_text": item.target_text,
                    "translate": id(item) in pending_ids,
                }
            )
        request_path = self.root / "request.json"
        response_path = self.root / "response.json"
        request_path.write_text(
            json.dumps(
                {
                    "model": self.model_name,
                    "source_language": SOURCE_LANGUAGE_NAMES.get(source_language.lower(), source_language),
                    "target_language": TARGET_LANGUAGE_NAMES[target_language],
                    "context_size": self.context_size,
                    "style": self.style,
                    "glossary": self.glossary,
                    "quantization": self.quantization,
                    "segments": payload_segments,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        worker = Path(__file__).resolve().parents[1] / "scripts" / "translation_worker.py"
        _run_visible(
            [str(self.python), "-u", str(worker), "--input", str(request_path), "--output", str(response_path)],
            label="HY-MT2-7B 上下文与时长感知翻译（首次会下载权重）",
            progress=progress,
            env=self._worker_env(),
        )
        response = json.loads(response_path.read_text(encoding="utf-8"))
        translations = {int(item["id"]): str(item["target_text"]).strip() for item in response["segments"]}
        for item in segments:
            translated = translations.get(item.id, "")
            if not translated:
                raise RuntimeError(f"HY-MT2 没有返回第 {item.id + 1} 行译文")
            item.target_text = translated
            item.target_language = target_language
        release_cuda()
        return segments


def translate_segments(
    segments: list[Segment],
    *,
    backend: str,
    source_language: str,
    target_language: str,
    model_name: str = "facebook/nllb-200-distilled-600M",
    api_base: str = "",
    api_key: str = "",
    api_model: str = "",
    runtime_root: str | None = None,
    context_size: int = 3,
    style: str = "自然、准确、符合人物身份的影视对白",
    glossary: str = "",
    hymt2_quantization: str = "auto",
    index_api_base: str = "http://127.0.0.1:8000/v1",
    index_api_key: str = "",
    index_model: str = "",
    index_syllables_per_second: float = 4.5,
    index_max_tokens: int = 1024,
    progress: Callable[[float, str], None] | None = None,
) -> list[Segment]:
    if not segments:
        return segments
    pending = [
        segment
        for segment in segments
        if not segment.translation_locked and not segment.target_text.strip()
    ]
    if not pending:
        if progress:
            progress(1.0, "没有需要翻译的行；人工译文已保留")
        return segments
    if backend == "none":
        for segment in pending:
            segment.target_text = segment.source_text
            segment.target_language = target_language
        return segments
    if backend == "nllb":
        translator = NLLBTranslator(model_name=model_name, device="auto")
    elif backend == "hymt2":
        translator = HyMT2Translator(
            model_name=model_name or "tencent/Hy-MT2-7B",
            runtime_root=runtime_root or str(runtime_paths().translation),
            quantization=hymt2_quantization,
            context_size=context_size,
            style=style,
            glossary=glossary,
        )
    elif backend == "llm":
        translator = ChatCompletionsTranslator(api_base, api_key, api_model)
    elif backend in {"index_public", "index", "homura"}:
        from .index_translation import (
            HOMURA_MODEL,
            PUBLIC_API_BASE,
            PUBLIC_MODEL,
            TRANSLATE_MODEL,
            IndexTranslator,
        )
        public = backend == "index_public"
        translator = IndexTranslator(
            api_base=PUBLIC_API_BASE if public else index_api_base,
            api_key="" if public else index_api_key,
            model=PUBLIC_MODEL if public else (index_model or (
                HOMURA_MODEL if backend == "homura" else TRANSLATE_MODEL
            )),
            homura=backend == "homura",
            context_size=context_size,
            style=style,
            glossary=glossary,
            syllables_per_second=index_syllables_per_second,
            max_tokens=index_max_tokens,
        )
    else:
        raise ValueError(f"未知翻译后端：{backend}")
    translator.translate(
        pending,
        source_language=source_language,
        target_language=target_language,
        context_segments=segments,
        progress=progress,
    )
    return segments
