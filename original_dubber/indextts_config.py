from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

V25_TEXT_TOKEN_COUNT = 60_509
V25_RESOURCE_VALUES = {
    "gpt_checkpoint": "gpt.pth",
    "s2mel_checkpoint": "s2mel.pth",
    "w2v_stat": "wav2vec2bert_stats.pt",
    "emo_matrix": "feat2.pt",
    "spk_matrix": "feat1.pt",
    "qwen_emo_path": "qwen0.6bemo4-merge/",
}


def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"无法解析 IndexTTS 配置 {path}：{exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"IndexTTS 配置不是 YAML 对象：{path}")
    return loaded


def _contains_internal_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_internal_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_internal_path(item) for item in value)
    return isinstance(value, str) and "/cubefs/" in value


def _architecture_is_v25(config: dict[str, Any]) -> bool:
    gpt = config.get("gpt")
    semantic_codec = config.get("semantic_codec")
    s2mel = config.get("s2mel")
    if (
        not isinstance(gpt, dict)
        or not isinstance(semantic_codec, dict)
        or not isinstance(s2mel, dict)
    ):
        return False
    condition = gpt.get("condition_module")
    emotion_condition = gpt.get("emo_condition_module")
    length_regulator = s2mel.get("length_regulator")
    return bool(
        gpt.get("number_text_tokens") == V25_TEXT_TOKEN_COUNT
        and gpt.get("condition_type") == "conformer_perceiver"
        and isinstance(condition, dict)
        and condition.get("input_layer") == "conv2d2"
        and isinstance(emotion_condition, dict)
        and emotion_condition.get("input_layer") == "conv2d2"
        and semantic_codec.get("codebook_size") == 8192
        and isinstance(length_regulator, dict)
        and length_regulator.get("in_channels") == 1024
    )


def inspect_indextts25_config(path: str | Path) -> dict[str, Any]:
    """Inspect active YAML values instead of matching comments as strings."""
    config = _load_config(path)
    architecture_is_v25 = _architecture_is_v25(config)
    version = config.get("version")
    normalized_version = str(version).strip() if version is not None else ""
    has_internal_path = _contains_internal_path(config)
    resource_mismatches = {
        key: config.get(key)
        for key, expected in V25_RESOURCE_VALUES.items()
        if config.get(key) != expected
    }
    vocoder = config.get("vocoder")
    if not isinstance(vocoder, dict) or vocoder.get("name") != "bigvgan_generator.pt":
        resource_mismatches["vocoder.name"] = (
            vocoder.get("name") if isinstance(vocoder, dict) else None
        )
    errors: list[str] = []
    if not architecture_is_v25:
        errors.append("缺少 IndexTTS 2.5 专属模型结构签名")
    if normalized_version != "2.5":
        errors.append(f"version 应为 2.5，当前为 {version!r}")
    if has_internal_path:
        errors.append("包含无法在 Colab 使用的 /cubefs 内部路径")
    if resource_mismatches:
        errors.append("模型资源路径不正确：" + ", ".join(resource_mismatches))
    return {
        "config_is_v25": not errors,
        "architecture_is_v25": architecture_is_v25,
        "version": version,
        "has_internal_path": has_internal_path,
        "resource_mismatches": resource_mismatches,
        "errors": errors,
    }


def normalize_indextts25_hub_config(path: str | Path) -> dict[str, Any]:
    """Repair the known bad active paths/version in the official 2.5 Hub config.

    The rewrite is permitted only after the 2.5 architecture signature matches,
    so a genuine IndexTTS 2.0 config can never be silently relabelled as 2.5.
    """
    path = Path(path)
    config = _load_config(path)
    if not _architecture_is_v25(config):
        raise ValueError(
            "下载的配置不具备 IndexTTS 2.5 专属结构，拒绝自动修复；"
            "请确认模型仓库是 IndexTeam/IndexTTS-2.5"
        )

    changes: dict[str, dict[str, Any]] = {}
    for key, expected in V25_RESOURCE_VALUES.items():
        current = config.get(key)
        if current != expected:
            changes[key] = {"from": current, "to": expected}
            config[key] = expected
    vocoder = config.setdefault("vocoder", {})
    if not isinstance(vocoder, dict):
        raise ValueError("下载的 2.5 配置中 vocoder 不是对象，拒绝自动修复")
    if vocoder.get("name") != "bigvgan_generator.pt":
        changes["vocoder.name"] = {
            "from": vocoder.get("name"),
            "to": "bigvgan_generator.pt",
        }
        vocoder["name"] = "bigvgan_generator.pt"
    if str(config.get("version", "")).strip() != "2.5":
        changes["version"] = {"from": config.get("version"), "to": 2.5}
        config["version"] = 2.5

    backup = path.with_name("config.hub-original.yaml")
    if changes:
        if not backup.exists():
            shutil.copy2(path, backup)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    report = inspect_indextts25_config(path)
    if not report["config_is_v25"]:
        raise ValueError("2.5 配置规范化后仍未通过：" + "；".join(report["errors"]))
    return {
        **report,
        "changes": changes,
        "backup": str(backup) if backup.exists() else "",
    }
