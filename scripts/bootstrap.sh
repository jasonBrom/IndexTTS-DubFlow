#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf '找不到 %s。请先安装 Python 3.11。\n' "${PYTHON_BIN}" >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/setup_runtime.py" "$@"

