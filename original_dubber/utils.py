from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class CommandError(RuntimeError):
    pass


def run_command(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if result.returncode:
        rendered = " ".join(args)
        raise CommandError(f"命令失败 ({result.returncode})：{rendered}\n{result.stdout or ''}")
    return result


def require_binary(name: str) -> None:
    from shutil import which

    if which(name) is None:
        raise RuntimeError(f"缺少系统命令 {name}，请先运行 Colab 安装单元。")


def safe_name(value: str, fallback: str = "project") -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", value.strip())
    value = value.strip("._-")
    return value[:80] or fallback


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def file_fingerprint(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    stat = path.stat()
    digest.update(str(stat.st_size).encode())
    with path.open("rb") as handle:
        if stat.st_size <= chunk_size * 8:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
        else:
            last_offset = max(0, stat.st_size - chunk_size)
            for index in range(8):
                handle.seek(round(last_offset * index / 7))
                digest.update(handle.read(chunk_size))
    return digest.hexdigest()[:20]


def stable_hash(parts: Iterable[Any]) -> str:
    raw = json.dumps(list(parts), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def seconds_to_srt(value: float) -> str:
    total_ms = max(0, int(round(value * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(path: str | Path, segments: Iterable[Any], translated: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    index = 1
    for segment in segments:
        if not getattr(segment, "enabled", True):
            continue
        text = segment.target_text if translated else segment.source_text
        text = str(text).strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n{seconds_to_srt(segment.start)} --> {seconds_to_srt(segment.end)}\n{text}\n"
        )
        index += 1
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def release_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
