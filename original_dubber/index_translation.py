"""Index-Translate / Homura clients; no local inference dependencies required."""
from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from .models import Segment

PUBLIC_API_BASE = "https://index-translate.bilibili.com/v1"
PUBLIC_MODEL = "Index-Translate-35B-A3B"
LOCAL_API_BASE = "http://127.0.0.1:8000/v1"
TRANSLATE_MODEL = "IndexTeam/Index-Translate-2B"
HOMURA_MODEL = "IndexTeam/Index-Homura-2B"
LANGUAGE_NAMES = {
    "zh": "中文", "en": "英语", "ja": "日语", "es": "西班牙语", "ar": "阿拉伯语",
    "de": "德语", "fr": "法语", "ko": "韩语", "pt": "葡萄牙语", "ru": "俄语",
    "vi": "越南语", "yue": "粤语",
}


def completion_endpoint(base: str) -> str:
    base = base.strip().rstrip("/")
    parts = urlsplit(base)
    if (parts.scheme not in {"http", "https"} or not parts.hostname
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError("Index API 地址需要是 http(s) URL，不能包含内嵌凭据、查询参数或片段")
    if base.endswith("/chat/completions"):
        return base
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")


def glossary_pairs(value: str) -> str:
    """Accept the existing UI's one pair per line or a JSON string dictionary."""
    if not value.strip():
        return ""
    if value.lstrip().startswith("{"):
        try:
            terms = json.loads(value)
        except ValueError as exc:
            raise ValueError("Index 术语表 JSON 格式错误") from exc
        if not isinstance(terms, dict) or not all(
            isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip()
            for k, v in terms.items()
        ):
            raise ValueError("Index 术语表 JSON 必须是非空词语到译名的字符串映射")
        return "、".join(f"{k} -> {v}" for k, v in terms.items())
    pairs = []
    for line in value.splitlines():
        if not line.strip():
            continue
        parts = re.split(r"\s*(?:->|→|=|：|:)\s*", line.strip(), maxsplit=1)
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ValueError("Index 术语表请每行填写 原词=译词，或填写 JSON 字典")
        pairs.append(" -> ".join(part.strip() for part in parts))
    return "、".join(pairs)


def translation_content(value: object) -> str:
    try:
        choice = value["choices"][0]
        reason = choice.get("finish_reason")
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError("Index API 返回结构错误：缺少译文") from exc
    if reason not in {None, "stop", "eos"}:
        raise RuntimeError(f"Index 翻译未正常完成（finish_reason={reason}），未保存不完整译文")
    if not isinstance(content, str):
        raise RuntimeError("Index API 未返回文本译文")
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    elif "<think>" in content:
        raise RuntimeError("Index API 只返回了未结束的思考内容，未保存为译文")
    content = content.strip()
    if not content:
        raise RuntimeError("Index API 返回空译文")
    return content


@dataclass(slots=True)
class IndexTranslator:
    api_base: str = LOCAL_API_BASE
    api_key: str = ""
    model: str = TRANSLATE_MODEL
    homura: bool = False
    context_size: int = 3
    style: str = "自然、准确、符合人物身份的影视对白"
    glossary: str = ""
    syllables_per_second: float = 4.5
    max_tokens: int = 1024
    timeout: float = 120
    retries: int = 2

    def __post_init__(self) -> None:
        completion_endpoint(self.api_base)
        if not self.model.strip():
            raise ValueError("Index 模型名不能为空")
        if not math.isfinite(self.syllables_per_second) or not 1 <= self.syllables_per_second <= 12:
            raise ValueError("Homura 每秒音节数必须在 1～12 之间")
        if not 128 <= self.max_tokens <= 8192:
            raise ValueError("Index 最大输出 token 数必须在 128～8192 之间")
        if not 0 <= self.context_size <= 12:
            raise ValueError("Index 翻译上下文句数必须在 0～12 之间")
        if not math.isfinite(self.timeout) or self.timeout <= 0 or not 0 <= self.retries <= 5:
            raise ValueError("Index 请求超时或重试次数无效")
        glossary_pairs(self.glossary)

    def _prompt(self, segment: Segment, timeline: list[Segment], source: str, target: str) -> str:
        target_name = LANGUAGE_NAMES.get(target.lower(), target)
        terms = glossary_pairs(self.glossary)
        constraints = []
        if self.style.strip():
            constraints.append(f"【注意】文风：{self.style.strip()}")
        position = next((i for i, item in enumerate(timeline) if item is segment), None)
        if position is not None and self.context_size:
            neighbors = timeline[max(0, position - self.context_size):position]
            following = timeline[position + 1:position + 1 + self.context_size]
            context = [
                {"位置": "前文", "说话人": item.speaker, "原文": item.source_text,
                 "译文": item.target_text}
                for item in neighbors
            ] + [
                {"位置": "后文", "说话人": item.speaker, "原文": item.source_text}
                for item in following
            ]
            if context:
                constraints.append("【注意】以下仅供消歧和统一称谓，不翻译或复述上下文："
                                   + json.dumps(context, ensure_ascii=False))
        text = segment.source_text.strip()
        if self.homura:
            if not math.isfinite(segment.duration) or segment.duration <= 0:
                raise ValueError(f"第 {segment.id + 1} 行缺少有效时长，无法估算音节数")
            budget = max(1, math.floor(segment.duration * self.syllables_per_second + 0.5))
            prompt = f"请将以下文本翻译为{target_name}，译文严格控制在 {budget} 个音节。"
            if terms:
                prompt += f"要求：严格遵守术语映射表【{terms}】。"
            if constraints:
                prompt += "\n" + "\n".join(constraints) + "\n"
            return prompt + f"直接输出翻译结果，不要进行任何解释。\n\n{text}"
        if terms:
            constraints.insert(0, f"【硬性要求】专名/术语对照: {terms}")
        constraints.append(f"【注意】对白时长约 {segment.duration:.2f} 秒，措辞适合自然朗读；保留语义、语气、姓名和数字")
        source_name = "" if source in {"", "auto"} else LANGUAGE_NAMES.get(source.lower(), source)
        requirements = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(constraints))
        return (f"请将以下{source_name}对白翻译成{target_name}，并且严格遵循所有约束要求。\n\n"
                f"【源文】\n{text}\n\n【约束要求】\n{requirements}\n\n只输出译文，不要有任何额外说明。")

    def _complete(self, prompt: str, progress: Callable[[float, str], None] | None, fraction: float) -> str:
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3 if self.homura else 0,
            "max_tokens": self.max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json", "User-Agent": "IndexTTS-DubFlow/0.3.0"}
        if self.api_key.strip():
            headers["Authorization"] = "Bearer " + self.api_key.strip()
        request = urllib.request.Request(
            completion_endpoint(self.api_base), data=json.dumps(payload).encode("utf-8"), headers=headers
        )
        for attempt in range(self.retries + 1):
            delay = min(2 ** attempt, 8)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    try:
                        value = json.load(response)
                    except (ValueError, UnicodeError) as exc:
                        raise RuntimeError("Index API 未返回有效 JSON") from exc
                return translation_content(value)
            except urllib.error.HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
                exc.close()
                if status not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(f"Index API 返回 HTTP {status}；请检查服务地址、模型名、密钥或服务限流") from None
                if retry_after.isdigit():
                    delay = min(30, max(delay, int(retry_after)))
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.retries:
                    raise RuntimeError("Index API 连接失败或超时；请检查翻译服务是否启动以及网络连接") from exc
            if progress:
                progress(fraction, f"Index 服务暂不可用，{delay} 秒后重试 {attempt + 1}/{self.retries}")
            time.sleep(delay)
        raise RuntimeError("Index 翻译请求失败")

    def translate(
        self, segments: list[Segment], *, source_language: str, target_language: str,
        context_segments: list[Segment] | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[Segment]:
        timeline = context_segments if context_segments is not None else segments
        pending = [s for s in segments if s.enabled and not s.translation_locked and not s.target_text.strip()]
        for index, segment in enumerate(pending):
            prompt = self._prompt(segment, timeline, source_language, target_language)
            if progress:
                progress(index / len(pending), f"Index 翻译 {index + 1}/{len(pending)} · {self.model}")
            translated = self._complete(prompt, progress, index / len(pending))
            segment.target_text = translated
            segment.target_language = target_language
            if progress:
                progress((index + 1) / len(pending), f"Index 翻译 {index + 1}/{len(pending)} 完成")
        return segments
