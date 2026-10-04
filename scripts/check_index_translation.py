"""Smoke-test the same Index client used by DubFlow, without loading ASR/TTS."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from original_dubber.index_translation import (
    HOMURA_MODEL,
    LOCAL_API_BASE,
    PUBLIC_API_BASE,
    PUBLIC_MODEL,
    TRANSLATE_MODEL,
    IndexTranslator,
)
from original_dubber.models import Segment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="你好，世界。今天天气不错。")
    parser.add_argument("--backend", choices=["public", "local", "homura"], default="local")
    parser.add_argument("--base-url", default=LOCAL_API_BASE)
    parser.add_argument("--model", default="")
    parser.add_argument("--source", default="zh")
    parser.add_argument("--target", choices=["ZH", "EN", "JA", "ES", "AR"], default="EN")
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--syllables-per-second", type=float, default=4.5)
    parser.add_argument("--glossary", default="")
    args = parser.parse_args()
    segment = Segment(0, 0, args.seconds, args.text)
    segment.validate()
    public = args.backend == "public"
    translator = IndexTranslator(
        api_base=PUBLIC_API_BASE if public else args.base_url,
        api_key="" if public else os.environ.get("INDEX_API_KEY", ""),
        model=PUBLIC_MODEL if public else (args.model or (
            HOMURA_MODEL if args.backend == "homura" else TRANSLATE_MODEL
        )),
        homura=args.backend == "homura", context_size=0,
        syllables_per_second=args.syllables_per_second, glossary=args.glossary,
    )
    translator.translate([segment], source_language=args.source, target_language=args.target)
    print(segment.target_text)


if __name__ == "__main__":
    main()
