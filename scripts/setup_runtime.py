from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from original_dubber.platforms import runtime_paths, venv_executable

PINNED_COMMIT = "ccd81054de9859faeb19b773fff0e2e1ae9e959e"


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run(label: str, command: list[str], *, env: dict[str, str], cwd: Path | None = None) -> None:
    started = time.monotonic()
    print(f"\n[setup] ▶ {label}\n[setup] $ {command_text(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)
    print(f"[setup] ✅ {label}（{time.monotonic() - started:.1f}s）", flush=True)


def require_binary(name: str) -> None:
    if shutil.which(name):
        return
    system = platform.system()
    hints = {
        "Windows": "使用 winget install Git.Git 和 winget install Gyan.FFmpeg，随后重开终端。",
        "Darwin": "使用 brew install git ffmpeg。",
        "Linux": "Debian/Ubuntu 可运行 sudo apt-get install -y git ffmpeg。",
    }
    raise RuntimeError(f"缺少系统命令 {name!r}。{hints.get(system, '请先安装并加入 PATH。')}")


def ensure_uv(env: dict[str, str]) -> list[str]:
    if importlib.util.find_spec("uv") is None:
        run("安装 uv", [sys.executable, "-m", "pip", "install", "-U", "uv"], env=env)
    return [sys.executable, "-m", "uv"]


def ensure_free_space(root: Path, minimum_gib: float, ignore: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 1024**3
    print(f"[setup] 运行目录可用空间：{free:.2f} GiB（建议至少 {minimum_gib:.0f} GiB）")
    if free < minimum_gib and not ignore:
        raise RuntimeError(
            f"可用空间只有 {free:.2f} GiB。完整主环境、IndexTTS 与辅助模型建议至少"
            f" {minimum_gib:.0f} GiB；请更换 --runtime-root，或显式添加 --ignore-space-check。"
        )


def prepare_source(index_dir: Path, env: dict[str, str]) -> None:
    if index_dir.exists() and any(index_dir.iterdir()) and not (index_dir / ".git").is_dir():
        raise RuntimeError(f"{index_dir} 非空且不是 Git 仓库，请换一个 --runtime-root 或 --index-dir")
    index_dir.mkdir(parents=True, exist_ok=True)
    if not (index_dir / ".git").is_dir():
        run("初始化 IndexTTS 仓库", ["git", "init"], env=env, cwd=index_dir)
        run(
            "配置 IndexTTS 上游",
            ["git", "remote", "add", "origin", "https://github.com/index-tts/index-tts.git"],
            env=env,
            cwd=index_dir,
        )
    remotes = subprocess.run(
        ["git", "remote"], cwd=index_dir, env=env, text=True, capture_output=True, check=True
    ).stdout.split()
    if "origin" not in remotes:
        run(
            "配置 IndexTTS 上游",
            ["git", "remote", "add", "origin", "https://github.com/index-tts/index-tts.git"],
            env=env,
            cwd=index_dir,
        )
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{PINNED_COMMIT}^{{commit}}"],
        cwd=index_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not exists:
        run(
            f"下载固定提交 {PINNED_COMMIT[:12]}",
            ["git", "fetch", "--depth", "1", "origin", PINNED_COMMIT],
            env=env,
            cwd=index_dir,
        )
    run("切换到固定提交", ["git", "checkout", "--detach", PINNED_COMMIT], env=env, cwd=index_dir)


def apply_patch(index_dir: Path, env: dict[str, str]) -> None:
    patch = PROJECT_ROOT / "patches" / "indextts25_exact_duration.patch"
    forward = subprocess.run(
        ["git", "apply", "--check", str(patch)], cwd=index_dir, env=env
    ).returncode == 0
    if forward:
        run("应用精确时长补丁", ["git", "apply", str(patch)], env=env, cwd=index_dir)
        return
    reverse = subprocess.run(
        ["git", "apply", "--check", "-R", str(patch)], cwd=index_dir, env=env
    ).returncode == 0
    if not reverse:
        raise RuntimeError("精确时长补丁既不能应用，也不是已应用状态；请检查上游目录是否被修改。")
    print("[setup] ✅ 精确时长补丁已存在。")


def parse_args() -> argparse.Namespace:
    defaults = runtime_paths()
    parser = argparse.ArgumentParser(
        description="跨平台安装 IndexTTS 2.5 原声翻译运行环境（Linux / Windows / macOS）"
    )
    parser.add_argument("--runtime-root", type=Path, default=defaults.root)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-source", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--with-demucs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-singing-detector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-diarization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-models", action="store_true", help="仅准备源码和 Python 环境")
    parser.add_argument("--min-free-gib", type=float, default=25.0)
    parser.add_argument("--ignore-space-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只显示解析后的路径和平台，不做修改")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = runtime_paths(args.runtime_root)
    index_dir = (args.index_dir or paths.indextts).expanduser().resolve()
    model_dir = (args.model_dir or index_dir / "checkpoints").expanduser().resolve()
    print(
        f"[setup] 平台：{platform.platform()}\n"
        f"[setup] 项目：{PROJECT_ROOT}\n"
        f"[setup] 运行目录：{paths.root}\n"
        f"[setup] IndexTTS：{index_dir}\n"
        f"[setup] 模型：{model_dir}"
    )
    if args.dry_run:
        return 0

    for binary in ("git", "ffmpeg", "ffprobe"):
        require_binary(binary)
    ensure_free_space(paths.root, args.min_free_gib, args.ignore_space_check)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "HF_HUB_DISABLE_XET": env.get("HF_HUB_DISABLE_XET", "1"),
            "UV_HTTP_TIMEOUT": env.get("UV_HTTP_TIMEOUT", "300"),
            "UV_LINK_MODE": env.get("UV_LINK_MODE", "copy" if os.name == "nt" else "hardlink"),
        }
    )
    uv = ensure_uv(env)
    prepare_source(index_dir, env)
    apply_patch(index_dir, env)

    run(
        "同步 IndexTTS 官方 webui 环境",
        [*uv, "sync", "--project", str(index_dir), "--frozen", "--extra", "webui", "--no-dev"],
        env=env,
    )
    runtime_python = venv_executable(index_dir / ".venv")
    if not runtime_python.is_file():
        raise RuntimeError(f"uv 已完成但未找到运行时 Python：{runtime_python}")

    packages = [
        "faster-whisper>=1.1,<2",
        "soundfile>=0.12,<1",
        "socksio>=1,<2",
        "PyYAML>=6,<7",
        "huggingface-hub>=0.34,<1",
    ]
    if args.model_source == "modelscope":
        packages.append("modelscope>=1.28,<2")
    if args.with_demucs:
        packages.append("demucs>=4,<5")
    if args.with_diarization:
        packages.append("pyannote.audio>=3.3,<4")
    run(
        "安装视频译制流水线依赖",
        [*uv, "pip", "install", "--python", str(runtime_python), *packages],
        env=env,
    )

    if not args.skip_models:
        download = [
            str(runtime_python),
            "-u",
            str(PROJECT_ROOT / "scripts" / "download_models.py"),
            "--model-dir",
            str(model_dir),
            "--source",
            args.model_source,
        ]
        if args.with_singing_detector:
            download.append("--include-singing")
        run("下载并校验模型", download, env=env)
        run(
            "规范化 IndexTTS 2.5 配置",
            [
                str(runtime_python),
                "-u",
                str(PROJECT_ROOT / "scripts" / "normalize_indextts25_config.py"),
                str(model_dir / "config.yaml"),
            ],
            env=env,
        )

        doctor_env = env.copy()
        doctor_env.update(
            {
                "DUBBER_RUNTIME_ROOT": str(paths.root),
                "INDEXTTS_DIR": str(index_dir),
                "INDEXTTS_MODEL_DIR": str(model_dir),
                "INDEXTTS_CONFIG": str(model_dir / "config.yaml"),
                "INSTALL_DEMUCS": "1" if args.with_demucs else "0",
                "INSTALL_SINGING_DETECTOR": "1" if args.with_singing_detector else "0",
                "INSTALL_DIARIZATION": "1" if args.with_diarization else "0",
            }
        )
        run(
            "最终自检",
            [str(runtime_python), "-u", str(PROJECT_ROOT / "scripts" / "doctor.py")],
            env=doctor_env,
        )

    if args.clean_cache:
        subprocess.run([*uv, "cache", "clean"], env=env, check=False)
        subprocess.run([str(runtime_python), "-m", "pip", "cache", "purge"], env=env, check=False)

    launcher = "scripts\\run.ps1" if os.name == "nt" else "scripts/run.sh"
    print(f"\n[setup] ✅ 安装完成。启动命令：{launcher}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

