from __future__ import annotations

import html
import json
import re
from pathlib import Path

from .models import Segment

_TIME_PATTERN = re.compile(
    r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2})(?P<f>[,.]\d{1,3})?"
)
_HTML_TAG = re.compile(r"<[^>]+>")
_ASS_TAG = re.compile(r"\{[^}]*\}")


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "big5"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _timestamp(value: str) -> float:
    match = _TIME_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"无法解析字幕时间：{value!r}")
    fraction = match.group("f") or ""
    milliseconds = int(fraction[1:].ljust(3, "0")) if fraction else 0
    return (
        int(match.group("h") or 0) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + milliseconds / 1000
    )


def _clean_text(value: str) -> str:
    value = value.replace("\\N", "\n").replace("\\n", "\n")
    value = _ASS_TAG.sub("", value)
    value = _HTML_TAG.sub("", value)
    value = html.unescape(value)
    return " ".join(part.strip() for part in value.splitlines() if part.strip()).strip()


def _parse_srt_or_vtt(text: str) -> list[tuple[float, float, str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text)
    cues: list[tuple[float, float, str]] = []
    timing = re.compile(
        r"(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
        r"(?P<end>(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{1,3})"
    )
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines()]
        timing_index = next((index for index, line in enumerate(lines) if timing.search(line)), None)
        if timing_index is None:
            continue
        match = timing.search(lines[timing_index])
        assert match is not None
        cue_text = _clean_text("\n".join(lines[timing_index + 1 :]))
        if cue_text:
            cues.append((_timestamp(match.group("start")), _timestamp(match.group("end")), cue_text))
    return cues


def _parse_ass(text: str) -> list[tuple[float, float, str]]:
    in_events = False
    fields = ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]
    cues: list[tuple[float, float, str]] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            in_events = line.lower() == "[events]"
            continue
        if not in_events:
            continue
        if line.lower().startswith("format:"):
            fields = [item.strip().lower() for item in line.split(":", 1)[1].split(",")]
            continue
        if not line.lower().startswith("dialogue:"):
            continue
        values = line.split(":", 1)[1].lstrip().split(",", max(0, len(fields) - 1))
        if len(values) < len(fields):
            continue
        item = dict(zip(fields, values, strict=False))
        cue_text = _clean_text(item.get("text", ""))
        if cue_text:
            cues.append((_timestamp(item["start"]), _timestamp(item["end"]), cue_text))
    return cues


def parse_subtitle_file(
    path: str | Path,
    *,
    mode: str,
    source_language: str,
    target_language: str,
) -> list[Segment]:
    subtitle_path = Path(path).expanduser()
    suffix = subtitle_path.suffix.lower()
    text = _read_text(subtitle_path)
    cues = _parse_ass(text) if suffix in {".ass", ".ssa"} else _parse_srt_or_vtt(text)
    if not cues:
        raise ValueError(f"字幕中没有找到有效时间轴：{subtitle_path.name}")
    segments: list[Segment] = []
    for start, end, cue_text in cues:
        if end <= start:
            continue
        translated = mode == "translated"
        segment = Segment(
            id=len(segments),
            start=start,
            end=end,
            source_text=cue_text,
            target_text=cue_text if translated else "",
            source_language=(target_language.lower() if translated else source_language),
            target_language=target_language,
            translation_locked=translated,
        )
        segment.validate()
        segments.append(segment)
    if not segments:
        raise ValueError("字幕时间均无效，请检查开始和结束时间")
    return segments


def import_reviewed_timeline(
    path: str | Path,
    current: list[Segment],
) -> list[Segment]:
    """Merge a reviewed JSON/SRT/VTT/ASS timeline into an analysed project.

    JSON preserves every editable field. Subtitle formats update translated
    text by cue index/time and lock those lines so a later analysis cannot
    overwrite the user's corrections.
    """
    review_path = Path(path).expanduser()
    suffix = review_path.suffix.lower()
    if suffix == ".json":
        value = json.loads(_read_text(review_path))
        raw_segments = value.get("segments") if isinstance(value, dict) else value
        if not isinstance(raw_segments, list):
            raise ValueError("时间轴 JSON 缺少 segments 数组")
        reviewed = [Segment.from_dict(item) for item in raw_segments]
        if len(reviewed) != len(current):
            raise ValueError(
                f"时间轴行数不一致：当前 {len(current)} 行，上传文件 {len(reviewed)} 行"
            )
        result: list[Segment] = []
        for index, (existing, incoming) in enumerate(
            zip(current, reviewed, strict=False)
        ):
            if abs(existing.start - incoming.start) > 0.25:
                raise ValueError(f"第 {index + 1} 行开始时间与当前项目不匹配")
            existing.enabled = incoming.enabled
            existing.start = incoming.start
            existing.end = incoming.end
            existing.speaker = incoming.speaker or existing.speaker
            existing.source_text = incoming.source_text or existing.source_text
            existing.target_text = incoming.target_text
            existing.translation_locked = bool(incoming.target_text.strip())
            existing.emotion_text = incoming.emotion_text
            existing.render_start = None
            existing.render_end = None
            existing.max_render_end = None
            existing.generated_path = ""
            existing.validate()
            result.append(existing)
        return result

    if suffix not in {".srt", ".vtt", ".ass", ".ssa"}:
        raise ValueError("校对时间轴只支持 JSON、SRT、VTT、ASS、SSA")
    text = _read_text(review_path)
    cues = _parse_ass(text) if suffix in {".ass", ".ssa"} else _parse_srt_or_vtt(text)
    if len(cues) != len(current):
        raise ValueError(
            f"字幕行数不一致：当前 {len(current)} 行，上传字幕 {len(cues)} 行；"
            "为防止译文错位，请先导出本项目时间轴再校对。"
        )
    for index, (segment, (start, end, text_value)) in enumerate(
        zip(current, cues, strict=False)
    ):
        if abs(segment.start - start) > 0.25:
            raise ValueError(f"第 {index + 1} 行时间与当前项目不匹配")
        segment.start = start
        segment.end = end
        segment.target_text = text_value.strip()
        segment.translation_locked = True
        segment.render_start = None
        segment.render_end = None
        segment.max_render_end = None
        segment.generated_path = ""
        segment.validate()
    return current
