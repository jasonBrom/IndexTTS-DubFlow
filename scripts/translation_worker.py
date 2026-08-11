from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

GIB = 1024**3
MODEL_WEIGHT_BYTES = {
    "tencent/Hy-MT2-7B": 16_059_080_704,
    # 1.8B is BF16 and is a little under 4 GiB including tokenizer/config files.
    "tencent/Hy-MT2-1.8B": 3_900_000_000,
}


def _allocated_bytes(path: Path) -> int:
    """Return allocated bytes once per inode, so Hub snapshot symlinks are not double-counted."""
    if not path.exists():
        return 0
    seen: set[tuple[int, int]] = set()
    total = 0
    for item in path.rglob("*"):
        try:
            stat = item.stat()
        except OSError:
            continue
        if not item.is_file():
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        blocks = getattr(stat, "st_blocks", 0)
        total += blocks * 512 if blocks else stat.st_size
    return total


def _model_cache_path(hf_home: Path, model_name: str) -> Path:
    return hf_home / "hub" / ("models--" + model_name.replace("/", "--"))


def prepare_model_storage(model_name: str) -> dict[str, float | str]:
    hf_home = Path(os.environ.get("HF_HOME") or "~/.cache/huggingface").expanduser().resolve()
    hf_home.mkdir(parents=True, exist_ok=True)

    removed_xet = 0
    xet_cache = hf_home / "xet"
    if os.environ.get("HF_HUB_DISABLE_XET", "1").lower() not in {"0", "false", "no"}:
        removed_xet = _allocated_bytes(xet_cache)
        if xet_cache.exists():
            shutil.rmtree(xet_cache)

    model_cache = _model_cache_path(hf_home, model_name)
    cached = _allocated_bytes(model_cache)
    expected = MODEL_WEIGHT_BYTES.get(model_name, 0)
    remaining = max(0, expected - cached)
    free = shutil.disk_usage(hf_home).free
    reserve = 3 * GIB
    required = remaining + reserve if expected else reserve
    report: dict[str, float | str] = {
        "hf_home": str(hf_home),
        "free_gib": free / GIB,
        "cached_gib": cached / GIB,
        "remaining_gib": remaining / GIB,
        "removed_xet_gib": removed_xet / GIB,
    }
    print(
        "[HY-MT2] 磁盘预检："
        f"HF_HOME={hf_home}；可用={report['free_gib']:.2f} GiB；"
        f"本模型已缓存={report['cached_gib']:.2f} GiB；"
        f"预计还需={report['remaining_gib']:.2f} GiB；"
        f"已回收 Xet 临时块={report['removed_xet_gib']:.2f} GiB",
        flush=True,
    )
    if free < required:
        short = (required - free) / GIB
        raise RuntimeError(
            f"HY-MT2 下载前磁盘预检未通过：至少还缺 {short:.2f} GiB。"
            "请运行 Notebook 的“磁盘诊断与安全清理”单元，或把 HYMT_CACHE_ON_DRIVE=True；"
            "若希望进一步节省空间，可在 Web 中改选 tencent/Hy-MT2-1.8B。"
        )
    return report


def _clean_translation(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:text)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(
        r"^(?:translation|译文|翻译结果)\s*[:：]\s*", "", value, flags=re.IGNORECASE
    )
    return value.strip().strip('"').strip()


def _prompt(payload: dict, index: int, previous_targets: dict[int, str]) -> str:
    segments = payload["segments"]
    item = segments[index]
    context_size = int(payload.get("context_size", 3))
    before = segments[max(0, index - context_size) : index]
    after = segments[index + 1 : index + 1 + min(2, context_size)]
    context_lines: list[str] = []
    for context_item in before:
        translated = previous_targets.get(int(context_item["id"])) or context_item.get("target_text")
        line = f'- 前文 {context_item["speaker"]}: {context_item["source_text"]}'
        if translated:
            line += f" -> {translated}"
        context_lines.append(line)
    for context_item in after:
        context_lines.append(
            f'- 后文 {context_item["speaker"]}: {context_item["source_text"]}'
        )
    glossary = str(payload.get("glossary") or "").strip()
    duration = float(item["duration"])
    return (
        f"将下面的影视对白从 {payload['source_language']} 翻译为 {payload['target_language']}。\n"
        "你必须只输出这一句的译文，不要解释、不要引号、不要标签。\n"
        "优先保证语义、人物关系、指代、专名、数字、语气和情感准确；同时使用目标语言自然口语。\n"
        f"译文将用于约 {duration:.2f} 秒的配音。把时长当作软约束：过长时自然精简，"
        "过短时可补全目标语言自然表达，但不得删改关键信息。\n"
        f"整体风格：{payload.get('style') or '自然影视对白'}\n"
        f"术语表：\n{glossary or '（无）'}\n"
        f"上下文：\n{chr(10).join(context_lines) or '（无）'}\n"
        f"当前说话人：{item['speaker']}\n"
        f"待翻译文本：{item['source_text']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))

    model_name = payload.get("model") or "tencent/Hy-MT2-7B"
    prepare_model_storage(model_name)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = payload.get("quantization", "auto")
    total_gib = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
        if torch.cuda.is_available()
        else 0.0
    )
    if not torch.cuda.is_available():
        raise RuntimeError("HY-MT2-7B 本地翻译需要 CUDA GPU；请改用 NLLB/API 或启用 GPU 运行时")
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    use_4bit = quantization == "4bit" or (quantization == "auto" and total_gib < 22)
    print(
        f"[HY-MT2] 加载 {model_name}；GPU={total_gib:.1f} GiB；"
        f"精度={'4bit' if use_4bit else 'bf16'}。"
        "注意：bitsandbytes 4-bit 节省显存，但 Hub 仍需下载原始权重。",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16_supported else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["dtype"] = torch.bfloat16 if bf16_supported else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).eval()
    print("[HY-MT2] 模型已加载，开始逐句上下文翻译", flush=True)

    previous_targets: dict[int, str] = {}
    output_segments: list[dict] = []
    targets = [item for item in payload["segments"] if item.get("translate")]
    completed = 0
    for index, item in enumerate(payload["segments"]):
        if not item.get("translate"):
            if item.get("target_text"):
                previous_targets[int(item["id"])] = str(item["target_text"])
            continue
        prompt = _prompt(payload, index, previous_targets)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                top_p=0.6,
                top_k=20,
                repetition_penalty=1.05,
                max_new_tokens=256,
            )
        new_tokens = generated[0, inputs.input_ids.shape[1] :]
        translated = _clean_translation(tokenizer.decode(new_tokens, skip_special_tokens=True))
        if not translated:
            raise RuntimeError(f"第 {int(item['id']) + 1} 行翻译结果为空")
        previous_targets[int(item["id"])] = translated
        output_segments.append({"id": int(item["id"]), "target_text": translated})
        completed += 1
        print(
            f"[HY-MT2] {completed}/{len(targets)} {item['speaker']} "
            f"{item['duration']:.2f}s -> {translated}",
            flush=True,
        )

    Path(args.output).write_text(
        json.dumps({"segments": output_segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[HY-MT2] 完成，共 {len(output_segments)} 行", flush=True)


if __name__ == "__main__":
    main()
