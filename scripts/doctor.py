from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from original_dubber.indextts_config import inspect_indextts25_config
from original_dubber.platforms import runtime_paths
from original_dubber.tts import MODEL_REQUIRED_FILES


def main() -> int:
    defaults = runtime_paths()
    index_dir = Path(os.environ.get("INDEXTTS_DIR", defaults.indextts))
    model_dir = Path(os.environ.get("INDEXTTS_MODEL_DIR", index_dir / "checkpoints"))
    config = Path(os.environ.get("INDEXTTS_CONFIG", model_dir / "config.yaml"))
    source = index_dir / "indextts" / "infer_v2_5.py"
    auxiliary_required = [
        "hf_cache/w2v-bert-2.0/config.json",
        "hf_cache/w2v-bert-2.0/preprocessor_config.json",
        "hf_cache/w2v-bert-2.0/model.safetensors",
        "hf_cache/bigvgan/config.json",
        "hf_cache/bigvgan/bigvgan_generator.pt",
        "hf_cache/campplus_cn_common.bin",
    ]
    required_modules = ["gradio", "faster_whisper", "soundfile", "huggingface_hub"]
    if os.environ.get("INSTALL_DEMUCS", "1") == "1":
        required_modules.append("demucs")
    if os.environ.get("INSTALL_DIARIZATION", "0") == "1":
        required_modules.append("pyannote.audio")

    def usable_file(relative: str) -> bool:
        path = model_dir / relative
        return path.is_file() and path.stat().st_size > 0

    def module_exists(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError):
            return False

    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "index_dir": str(index_dir),
        "model_dir": str(model_dir),
        "config": str(config),
        "source_exists": source.is_file(),
        "config_exists": config.is_file(),
        "missing_model_files": [name for name in MODEL_REQUIRED_FILES if not usable_file(name)],
        "missing_auxiliary_files": [
            name for name in auxiliary_required if not usable_file(name)
        ],
        "missing_python_modules": [
            name for name in required_modules if not module_exists(name)
        ],
        "qwen_emotion_ready": usable_file("qwen0.6bemo4-merge/config.json"),
        "exact_duration_patch": False,
        "config_is_v25": False,
        "config_architecture_is_v25": False,
        "config_version": None,
        "config_has_internal_path": False,
        "config_resource_mismatches": {},
        "config_errors": [],
        "singing_detector_required": os.environ.get("INSTALL_SINGING_DETECTOR", "1") == "1",
        "singing_detector_ready": (
            usable_file("hf_cache/ast-audioset/config.json")
            and usable_file("hf_cache/ast-audioset/preprocessor_config.json")
            and usable_file("hf_cache/ast-audioset/model.safetensors")
        ),
    }
    if source.is_file():
        raw = source.read_text(encoding="utf-8", errors="ignore")
        result["exact_duration_patch"] = "target_duration_seconds" in raw and "do_sample=do_sample" in raw
    if config.is_file():
        try:
            config_report = inspect_indextts25_config(config)
        except ValueError as exc:
            result["config_errors"] = [str(exc)]
        else:
            result["config_is_v25"] = config_report["config_is_v25"]
            result["config_architecture_is_v25"] = config_report["architecture_is_v25"]
            result["config_version"] = config_report["version"]
            result["config_has_internal_path"] = config_report["has_internal_path"]
            result["config_resource_mismatches"] = config_report["resource_mismatches"]
            result["config_errors"] = config_report["errors"]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(index_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    result["commit"] = commit
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:
        result["torch_error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    failed = bool(
        not result["source_exists"]
        or not result["config_exists"]
        or result["missing_model_files"]
        or result["missing_auxiliary_files"]
        or result["missing_python_modules"]
        or not result["qwen_emotion_ready"]
        or not result["exact_duration_patch"]
        or not result["config_is_v25"]
        or result["config_has_internal_path"]
        or (
            result["singing_detector_required"]
            and not result["singing_detector_ready"]
        )
        or commit != "ccd81054de9859faeb19b773fff0e2e1ae9e959e"
    )
    if failed:
        print("\n自检未通过，请不要启动模型。", file=sys.stderr)
        return 1
    if not result.get("cuda"):
        print("\n警告：没有 CUDA GPU，完整视频在 CPU 上会非常慢。")
    elif "T4" in str(result.get("gpu", "")):
        print("\n提示：T4 不支持 BF16，应用会自动改用 FP32；建议优先选择 L4/A100 运行时。")
    print("\nIndexTTS 2.5 代码、模型清单、配置和精确时长补丁自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
