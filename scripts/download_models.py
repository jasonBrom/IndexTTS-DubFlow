from __future__ import annotations

import argparse
import os
from pathlib import Path

MAIN_REQUIRED = (
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "wav2vec2bert_stats.pt",
    "feat1.pt",
    "feat2.pt",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "qwen0.6bemo4-merge/config.json",
)


def complete(root: Path, required: tuple[str, ...]) -> bool:
    return all((root / name).is_file() and (root / name).stat().st_size > 0 for name in required)


def hf_download(
    repo_id: str,
    destination: Path,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def download_main(source: str, model_dir: Path) -> None:
    if complete(model_dir, MAIN_REQUIRED):
        print(f"[models] IndexTTS 2.5 主模型已齐全：{model_dir}", flush=True)
        return
    print(f"[models] 下载 IndexTTS 2.5 主模型 → {model_dir}", flush=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if source == "modelscope":
        from modelscope import snapshot_download

        snapshot_download("IndexTeam/IndexTTS-2.5", local_dir=str(model_dir))
    else:
        hf_download("IndexTeam/IndexTTS-2.5", model_dir)
    missing = [name for name in MAIN_REQUIRED if not (model_dir / name).is_file()]
    if missing:
        raise RuntimeError("主模型下载不完整：" + ", ".join(missing))


def download_auxiliary(model_dir: Path, include_singing: bool) -> None:
    cache = model_dir / "hf_cache"
    tasks: list[tuple[str, Path, list[str], list[str] | None]] = [
        (
            "facebook/w2v-bert-2.0",
            cache / "w2v-bert-2.0",
            ["config.json", "preprocessor_config.json", "model.safetensors"],
            ["pytorch_model.bin"],
        ),
        (
            "nvidia/bigvgan_v2_22khz_80band_256x",
            cache / "bigvgan",
            ["config.json", "bigvgan_generator.pt"],
            None,
        ),
        (
            "funasr/campplus",
            cache,
            ["campplus_cn_common.bin"],
            None,
        ),
    ]
    if include_singing:
        tasks.append(
            (
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                cache / "ast-audioset",
                ["config.json", "preprocessor_config.json", "model.safetensors"],
                ["pytorch_model.bin"],
            )
        )
    for repo_id, destination, required, ignored in tasks:
        required_tuple = tuple(required)
        if complete(destination, required_tuple):
            print(f"[models] 缓存命中：{repo_id}", flush=True)
            continue
        print(f"[models] 下载辅助模型：{repo_id}", flush=True)
        hf_download(
            repo_id,
            destination,
            allow_patterns=required,
            ignore_patterns=ignored,
        )
        if not complete(destination, required_tuple):
            raise RuntimeError(f"辅助模型下载不完整：{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 IndexTTS 2.5 与译制辅助模型")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--include-singing", action="store_true")
    parser.add_argument("--skip-main", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    model_dir = args.model_dir.expanduser().resolve()
    if not args.skip_main:
        download_main(args.source, model_dir)
    download_auxiliary(model_dir, args.include_singing)
    print("[models] 所需模型文件已准备完成。", flush=True)


if __name__ == "__main__":
    main()

