from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def is_colab() -> bool:
    return bool(
        os.environ.get("COLAB_RELEASE_TAG")
        or os.environ.get("COLAB_BACKEND_VERSION")
        or os.environ.get("COLAB_GPU")
    )


def runtime_root() -> Path:
    configured = os.environ.get("DUBBER_RUNTIME_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if is_colab():
        return Path("/content/IndexTTS25_Runtime")
    return (PROJECT_ROOT / ".runtime").resolve()


def output_root() -> Path:
    configured = os.environ.get("DUBBER_OUTPUT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    drive = Path("/content/drive/MyDrive")
    if is_colab() and drive.is_dir():
        return drive / "IndexTTS_Dubber_Outputs"
    if is_colab():
        return Path("/content/IndexTTS_Dubber_Outputs")
    return Path.home() / "Videos" / "IndexTTS_Dubber_Outputs"


def venv_executable(
    venv: str | Path, executable: str = "python", *, windows: bool | None = None
) -> Path:
    root = Path(venv)
    use_windows_layout = os.name == "nt" if windows is None else windows
    if use_windows_layout:
        suffix = ".exe" if not executable.lower().endswith(".exe") else ""
        return root / "Scripts" / f"{executable}{suffix}"
    return root / "bin" / executable


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    root: Path
    indextts: Path
    checkpoints: Path
    asr: Path
    translation: Path
    cache: Path


def runtime_paths(root: str | Path | None = None) -> RuntimePaths:
    base = Path(root).expanduser().resolve() if root else runtime_root()
    indextts = base / "index-tts"
    return RuntimePaths(
        root=base,
        indextts=indextts,
        checkpoints=indextts / "checkpoints",
        asr=base / "asr-runtimes",
        translation=base / "translation-runtimes",
        cache=base / "cache",
    )
