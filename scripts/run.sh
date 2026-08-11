#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${DUBBER_RUNTIME_ROOT:-${PROJECT_ROOT}/.runtime}"
INDEXTTS_DIR="${INDEXTTS_DIR:-${RUNTIME_ROOT}/index-tts}"
INDEXTTS_MODEL_DIR="${INDEXTTS_MODEL_DIR:-${INDEXTTS_DIR}/checkpoints}"
PYTHON_BIN="${INDEXTTS_DIR}/.venv/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  printf '运行环境不存在：%s\n请先执行 scripts/bootstrap.sh。\n' "${PYTHON_BIN}" >&2
  exit 1
fi

export DUBBER_RUNTIME_ROOT="${RUNTIME_ROOT}"
export INDEXTTS_DIR INDEXTTS_MODEL_DIR
export INDEXTTS_CONFIG="${INDEXTTS_CONFIG:-${INDEXTTS_MODEL_DIR}/config.yaml}"
export ASR_RUNTIME_ROOT="${ASR_RUNTIME_ROOT:-${RUNTIME_ROOT}/asr-runtimes}"
export TRANSLATION_RUNTIME_ROOT="${TRANSLATION_RUNTIME_ROOT:-${RUNTIME_ROOT}/translation-runtimes}"
export HF_HOME="${HF_HOME:-${RUNTIME_ROOT}/cache/huggingface}"
export HYMT_HF_HOME="${HYMT_HF_HOME:-${HF_HOME}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1

exec "${PYTHON_BIN}" -u "${PROJECT_ROOT}/app.py" "$@"

