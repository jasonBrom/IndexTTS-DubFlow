#!/usr/bin/env bash
set -Eeuo pipefail

case "${1:-web}" in
  setup)
    shift
    exec /app/scripts/bootstrap.sh --runtime-root "${DUBBER_RUNTIME_ROOT:-/runtime}" "$@"
    ;;
  doctor)
    shift
    runtime_root="${DUBBER_RUNTIME_ROOT:-/runtime}"
    python_bin="${INDEXTTS_DIR:-${runtime_root}/index-tts}/.venv/bin/python"
    exec "${python_bin}" /app/scripts/doctor.py "$@"
    ;;
  web)
    shift
    runtime_root="${DUBBER_RUNTIME_ROOT:-/runtime}"
    python_bin="${INDEXTTS_DIR:-${runtime_root}/index-tts}/.venv/bin/python"
    if [ ! -x "${python_bin}" ]; then
      if [ "${AUTO_SETUP:-1}" != "1" ]; then
        printf '运行环境不存在，先执行 docker compose run --rm dubber setup。\n' >&2
        exit 1
      fi
      /app/scripts/bootstrap.sh \
        --runtime-root "${runtime_root}" \
        --model-source "${MODEL_SOURCE:-modelscope}"
    fi
    exec /app/scripts/run.sh "$@"
    ;;
  *)
    exec "$@"
    ;;
esac

