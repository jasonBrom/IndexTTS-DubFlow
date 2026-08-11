#!/usr/bin/env bash
set -Eeuo pipefail

PINNED_COMMIT="ccd81054de9859faeb19b773fff0e2e1ae9e959e"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEXTTS_DIR="${INDEXTTS_DIR:-/content/index-tts}"
INDEXTTS_MODEL_DIR="${INDEXTTS_MODEL_DIR:-${INDEXTTS_DIR}/checkpoints}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
INSTALL_DEMUCS="${INSTALL_DEMUCS:-1}"
INSTALL_DIARIZATION="${INSTALL_DIARIZATION:-0}"
INSTALL_SINGING_DETECTOR="${INSTALL_SINGING_DETECTOR:-1}"
CLEAN_INSTALLER_CACHE="${CLEAN_INSTALLER_CACHE:-1}"
LOG_HEARTBEAT_SECONDS="${LOG_HEARTBEAT_SECONDS:-10}"
TOTAL_STAGES=10
CURRENT_STAGE=0
CURRENT_STAGE_NAME="启动"
SETUP_STARTED_AT="${SECONDS}"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING="utf-8"
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export UV_HTTP_TIMEOUT="300"
export HF_HUB_DISABLE_PROGRESS_BARS=0
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export GIT_LFS_SKIP_SMUDGE=1

if ! [[ "${LOG_HEARTBEAT_SECONDS}" =~ ^[0-9]+$ ]] || \
   [ "${LOG_HEARTBEAT_SECONDS}" -lt 5 ] || \
   [ "${LOG_HEARTBEAT_SECONDS}" -gt 60 ]; then
  printf 'LOG_HEARTBEAT_SECONDS 必须是 5～60 的整数，当前为：%s\n' \
    "${LOG_HEARTBEAT_SECONDS}" >&2
  exit 2
fi

timestamp() {
  date '+%H:%M:%S'
}

format_seconds() {
  local total="${1:-0}"
  printf '%dm%02ds' "$((total / 60))" "$((total % 60))"
}

directory_size() {
  local path="${1:-}"
  if [ -n "${path}" ] && [ -e "${path}" ]; then
    du -sh "${path}" 2>/dev/null | awk '{print $1}' || printf '?'
  else
    printf '0B'
  fi
}

stage() {
  CURRENT_STAGE=$((CURRENT_STAGE + 1))
  CURRENT_STAGE_NAME="$1"
  printf '\n[%s] [%d/%d] %s\n' "$(timestamp)" "${CURRENT_STAGE}" "${TOTAL_STAGES}" "${CURRENT_STAGE_NAME}"
}

on_error() {
  local code=$?
  printf '\n[%s] ❌ 第 %d/%d 阶段失败：%s（退出码 %d）\n' \
    "$(timestamp)" "${CURRENT_STAGE}" "${TOTAL_STAGES}" "${CURRENT_STAGE_NAME}" "${code}" >&2
  printf '可以直接重新运行本单元；Git、uv 和模型下载器会复用已完成内容并续传。\n' >&2
  exit "${code}"
}
trap on_error ERR

run_visible() {
  local label="$1"
  local watch_path="$2"
  shift 2
  local started="${SECONDS}"
  printf '[%s] ▶ %s\n' "$(timestamp)" "${label}"
  "$@" &
  local command_pid=$!
  (
    while kill -0 "${command_pid}" 2>/dev/null; do
      sleep "${LOG_HEARTBEAT_SECONDS}"
      if kill -0 "${command_pid}" 2>/dev/null; then
        local elapsed=$((SECONDS - started))
        if [ -n "${watch_path}" ]; then
          printf '[%s] ⏳ %s，已运行 %s，当前目录 %s\n' \
            "$(timestamp)" "${label}" "$(format_seconds "${elapsed}")" "$(directory_size "${watch_path}")"
        else
          printf '[%s] ⏳ %s，已运行 %s\n' \
            "$(timestamp)" "${label}" "$(format_seconds "${elapsed}")"
        fi
      fi
    done
  ) &
  local heartbeat_pid=$!
  set +e
  wait "${command_pid}"
  local status=$?
  kill "${heartbeat_pid}" 2>/dev/null || true
  wait "${heartbeat_pid}" 2>/dev/null || true
  set -e
  if [ "${status}" -ne 0 ]; then
    printf '[%s] ❌ %s 失败（退出码 %d）\n' "$(timestamp)" "${label}" "${status}" >&2
    return "${status}"
  fi
  printf '[%s] ✅ %s，耗时 %s\n' \
    "$(timestamp)" "${label}" "$(format_seconds "$((SECONDS - started))")"
}

stage "检查系统工具"
printf 'IndexTTS 目录：%s\n模型目录：%s\n主模型来源：%s\n' \
  "${INDEXTTS_DIR}" "${INDEXTTS_MODEL_DIR}" "${MODEL_SOURCE}"
printf '磁盘空间：\n'
df -h /content 2>/dev/null | tail -n 1 || df -h . | tail -n 1
SYSTEM_PACKAGES=()
command -v git >/dev/null 2>&1 || SYSTEM_PACKAGES+=(git)
command -v ffmpeg >/dev/null 2>&1 || SYSTEM_PACKAGES+=(ffmpeg)
if [ "${#SYSTEM_PACKAGES[@]}" -gt 0 ]; then
  run_visible "更新 apt 索引" "" apt-get update
  run_visible "安装系统工具：${SYSTEM_PACKAGES[*]}" "" apt-get install -y "${SYSTEM_PACKAGES[@]}"
else
  printf '[%s] ✅ git 与 ffmpeg 已存在，跳过 apt。\n' "$(timestamp)"
fi

stage "准备 uv 包管理器"
if command -v uv >/dev/null 2>&1; then
  printf '[%s] ✅ 复用 %s\n' "$(timestamp)" "$(uv --version)"
else
  run_visible "安装 uv" "" python3 -m pip install -U uv
fi

stage "获取官方 main 中正式发布的 IndexTTS 2.5 源码"
if [ ! -d "${INDEXTTS_DIR}/.git" ]; then
  mkdir -p "${INDEXTTS_DIR}"
  git -C "${INDEXTTS_DIR}" init
  git -C "${INDEXTTS_DIR}" remote add origin https://github.com/index-tts/index-tts.git
fi
if ! git -C "${INDEXTTS_DIR}" remote get-url origin >/dev/null 2>&1; then
  git -C "${INDEXTTS_DIR}" remote add origin https://github.com/index-tts/index-tts.git
fi
if ! git -C "${INDEXTTS_DIR}" cat-file -e "${PINNED_COMMIT}^{commit}" 2>/dev/null; then
  run_visible \
    "下载固定提交 ${PINNED_COMMIT:0:12}" \
    "${INDEXTTS_DIR}/.git" \
    env GIT_LFS_SKIP_SMUDGE=1 git -C "${INDEXTTS_DIR}" fetch --depth 1 origin "${PINNED_COMMIT}"
else
  printf '[%s] ✅ 固定提交已在本地，跳过 Git 下载。\n' "$(timestamp)"
fi
env GIT_LFS_SKIP_SMUDGE=1 git -C "${INDEXTTS_DIR}" checkout --detach "${PINNED_COMMIT}"
printf '[%s] ✅ 源码版本：%s\n' "$(timestamp)" "$(git -C "${INDEXTTS_DIR}" rev-parse --short=12 HEAD)"

stage "应用确定性采样与自适应时长补丁"
if git -C "${INDEXTTS_DIR}" apply --check "${PROJECT_ROOT}/patches/indextts25_exact_duration.patch" 2>/dev/null; then
  git -C "${INDEXTTS_DIR}" apply "${PROJECT_ROOT}/patches/indextts25_exact_duration.patch"
  printf '[%s] ✅ 补丁已应用。\n' "$(timestamp)"
elif git -C "${INDEXTTS_DIR}" apply --check -R "${PROJECT_ROOT}/patches/indextts25_exact_duration.patch" 2>/dev/null; then
  printf '[%s] ✅ 补丁之前已经应用，直接复用。\n' "$(timestamp)"
else
  printf 'IndexTTS 精确时长补丁既不能应用，也不是已应用状态。请删除 %s 后重跑。\n' \
    "${INDEXTTS_DIR}" >&2
  exit 1
fi

stage "创建 Python 3.11 / CUDA 独立环境"
printf '这一阶段首次会下载 PyTorch 与 CUDA 运行库，通常是环境安装中最慢的一步。\n'
run_visible \
  "uv sync（仅 webui extra，不安装 DeepSpeed/FlashAttention）" \
  "${INDEXTTS_DIR}/.venv" \
  uv sync --project "${INDEXTTS_DIR}" --frozen --extra webui --no-dev
UV_PYTHON="${INDEXTTS_DIR}/.venv/bin/python"

stage "安装视频译制流水线依赖"
RUNTIME_PACKAGES=(
  "faster-whisper>=1.1,<2"
  "soundfile>=0.12,<1"
  "socksio>=1,<2"
  "huggingface-hub[cli,hf_xet]>=0.34,<1"
)
if [ "${INSTALL_DEMUCS}" = "1" ]; then
  RUNTIME_PACKAGES+=("demucs>=4.0,<5")
fi
if [ "${INSTALL_DIARIZATION}" = "1" ]; then
  RUNTIME_PACKAGES+=("pyannote.audio>=3.3,<4")
fi
run_visible \
  "安装 Whisper / 下载器及所选可选组件" \
  "${INDEXTTS_DIR}/.venv" \
  uv pip install --python "${UV_PYTHON}" "${RUNTIME_PACKAGES[@]}"

stage "下载 IndexTTS 2.5 主模型"
mkdir -p "${INDEXTTS_MODEL_DIR}"
MODEL_COMPLETE=1
for MODEL_FILE in \
  config.yaml gpt.pth s2mel.pth codec.pth wav2vec2bert_stats.pt feat1.pt feat2.pt \
  multilingual_zh_ja_yue_char_del.tiktoken; do
  if [ ! -s "${INDEXTTS_MODEL_DIR}/${MODEL_FILE}" ]; then
    MODEL_COMPLETE=0
    break
  fi
done
if [ ! -s "${INDEXTTS_MODEL_DIR}/qwen0.6bemo4-merge/config.json" ]; then
  MODEL_COMPLETE=0
fi
if [ "${MODEL_COMPLETE}" = "0" ]; then
  if [ "${MODEL_SOURCE}" = "modelscope" ]; then
    run_visible \
      "ModelScope IndexTTS 2.5 主模型与 QwenEmotion" \
      "${INDEXTTS_MODEL_DIR}" \
      uv run --project "${INDEXTTS_DIR}" modelscope download \
        --model IndexTeam/IndexTTS-2.5 \
        --local_dir "${INDEXTTS_MODEL_DIR}"
  elif [ "${MODEL_SOURCE}" = "huggingface" ]; then
    run_visible \
      "Hugging Face IndexTTS 2.5 主模型与 QwenEmotion" \
      "${INDEXTTS_MODEL_DIR}" \
      "${INDEXTTS_DIR}/.venv/bin/hf" download IndexTeam/IndexTTS-2.5 \
        --local-dir "${INDEXTTS_MODEL_DIR}"
  else
    printf 'MODEL_SOURCE 只支持 modelscope 或 huggingface，当前为：%s\n' "${MODEL_SOURCE}" >&2
    exit 1
  fi
else
  printf '[%s] ✅ 主模型文件齐全，跳过下载（%s）。\n' \
    "$(timestamp)" "$(directory_size "${INDEXTTS_MODEL_DIR}")"
fi
run_visible \
  "验证并规范化 IndexTTS 2.5 配置" \
  "" \
  "${UV_PYTHON}" -u "${PROJECT_ROOT}/scripts/normalize_indextts25_config.py" \
    "${INDEXTTS_MODEL_DIR}/config.yaml"

stage "下载推理辅助模型"
HF_CACHE="${INDEXTTS_MODEL_DIR}/hf_cache"
mkdir -p "${HF_CACHE}/w2v-bert-2.0" "${HF_CACHE}/bigvgan"
HF_BIN="${INDEXTTS_DIR}/.venv/bin/hf"

if [ ! -s "${HF_CACHE}/w2v-bert-2.0/config.json" ] || \
   [ ! -s "${HF_CACHE}/w2v-bert-2.0/preprocessor_config.json" ] || \
   [ ! -s "${HF_CACHE}/w2v-bert-2.0/model.safetensors" ]; then
  run_visible \
    "W2V-BERT 2.0" \
    "${HF_CACHE}/w2v-bert-2.0" \
    "${HF_BIN}" download facebook/w2v-bert-2.0 \
      --exclude "pytorch_model.bin" \
      --local-dir "${HF_CACHE}/w2v-bert-2.0"
else
  printf '[%s] ✅ W2V-BERT 缓存命中。\n' "$(timestamp)"
fi
if [ ! -s "${HF_CACHE}/bigvgan/config.json" ] || \
   [ ! -s "${HF_CACHE}/bigvgan/bigvgan_generator.pt" ]; then
  run_visible \
    "BigVGAN" \
    "${HF_CACHE}/bigvgan" \
    "${HF_BIN}" download nvidia/bigvgan_v2_22khz_80band_256x \
      config.json bigvgan_generator.pt \
      --local-dir "${HF_CACHE}/bigvgan"
else
  printf '[%s] ✅ BigVGAN 缓存命中。\n' "$(timestamp)"
fi
if [ ! -s "${HF_CACHE}/campplus_cn_common.bin" ]; then
  run_visible \
    "CAMPPlus 说话人编码器" \
    "${HF_CACHE}" \
    "${HF_BIN}" download funasr/campplus campplus_cn_common.bin --local-dir "${HF_CACHE}"
else
  printf '[%s] ✅ CAMPPlus 缓存命中。\n' "$(timestamp)"
fi
if [ "${INSTALL_SINGING_DETECTOR}" = "1" ] && \
   { [ ! -s "${HF_CACHE}/ast-audioset/config.json" ] || \
     [ ! -s "${HF_CACHE}/ast-audioset/preprocessor_config.json" ] || \
     [ ! -s "${HF_CACHE}/ast-audioset/model.safetensors" ]; }; then
  run_visible \
    "AST 对白/唱歌检测器" \
    "${HF_CACHE}/ast-audioset" \
    "${HF_BIN}" download MIT/ast-finetuned-audioset-10-10-0.4593 \
      --exclude "pytorch_model.bin" \
      --local-dir "${HF_CACHE}/ast-audioset"
elif [ "${INSTALL_SINGING_DETECTOR}" = "1" ]; then
  printf '[%s] ✅ AST 唱歌检测器缓存命中。\n' "$(timestamp)"
else
  printf '[%s] ℹ️ 未启用唱歌检测器，跳过 AST。\n' "$(timestamp)"
fi

stage "最终自检"
export INDEXTTS_DIR INDEXTTS_MODEL_DIR INSTALL_SINGING_DETECTOR INSTALL_DEMUCS INSTALL_DIARIZATION
export INDEXTTS_CONFIG="${INDEXTTS_MODEL_DIR}/config.yaml"
run_visible "检查代码、补丁、模型清单、CUDA 与显存" "" \
  "${UV_PYTHON}" -u "${PROJECT_ROOT}/scripts/doctor.py"

stage "回收安装器缓存"
if [ "${CLEAN_INSTALLER_CACHE}" = "1" ]; then
  UV_CACHE_PATH="${UV_CACHE_DIR:-$(uv cache dir 2>/dev/null || true)}"
  BEFORE_CACHE="$(directory_size "${UV_CACHE_PATH}")"
  uv cache clean >/dev/null 2>&1 || true
  "${UV_PYTHON}" -m pip cache purge >/dev/null 2>&1 || true
  printf '[%s] ✅ 已回收 uv/pip 安装缓存（清理前 uv 缓存 %s）；已安装环境不受影响。\n' \
    "$(timestamp)" "${BEFORE_CACHE}"
else
  printf '[%s] ℹ️ CLEAN_INSTALLER_CACHE=0，保留安装器缓存。\n' "$(timestamp)"
fi

printf '\n[%s] ✅ 全部安装完成，总耗时 %s。\n' \
  "$(timestamp)" "$(format_seconds "$((SECONDS - SETUP_STARTED_AT))")"
printf '再次运行会跳过已完成的源码与模型下载。现在请运行 Notebook 的“启动 Web 面板”单元。\n'
