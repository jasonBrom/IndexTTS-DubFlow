from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from original_dubber.indextts_config import normalize_indextts25_hub_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        report = normalize_indextts25_hub_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"IndexTTS 2.5 配置修复失败：{exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["changes"]:
        print("已确认 2.5 模型结构并修复官方 Hub 配置中的错误版本与内部路径。")
    else:
        print("IndexTTS 2.5 配置已经规范，无需修改。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
