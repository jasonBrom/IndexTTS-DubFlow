<div align="center">

# IndexTTS-DubFlow

**An original-voice video translation and intelligent dubbing workflow powered by IndexTTS 2.5**

Dialogue separation, speech recognition, contextual translation, voice and emotion cloning, timeline alignment, and final video delivery.

<!--
Keep external images on trusted HTTPS origins. GitHub automatically proxies and caches them
through GitHub Camo when rendering this README. Do not commit rendered camo.githubusercontent.com
hash URLs because they are not portable source links.
-->

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jasonBrom/IndexTTS-DubFlow/blob/main/notebooks/IndexTTS_DubFlow_Colab_R7.ipynb)
[![CI](https://github.com/jasonBrom/IndexTTS-DubFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/jasonBrom/IndexTTS-DubFlow/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/jasonBrom/IndexTTS-DubFlow)](https://github.com/jasonBrom/IndexTTS-DubFlow/releases)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB)](https://www.python.org/)
[![License](https://img.shields.io/github/license/jasonBrom/IndexTTS-DubFlow)](LICENSE)

[中文说明](README.md) · [Installation](docs/INSTALLATION.md) · [Configuration](docs/CONFIGURATION.md) · [Architecture](docs/ARCHITECTURE.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)

</div>

> [!IMPORTANT]
> This is not an official IndexTTS project. Obtain all necessary rights for the source video, voices, subtitles, translations, and generated output. Do not use it for fraud, impersonation, infringement, or unlawful activity.

## Overview

IndexTTS-DubFlow is an end-to-end video localization pipeline. It separates dialogue and background audio, transcribes and translates speech, reproduces each speaker's timbre and emotion, prioritizes natural speech timing, and produces a dubbed video together with editable timelines, subtitles, isolated audio, and quality-control artifacts.

Current release: `v0.2.0 / R7`  
Pinned upstream IndexTTS 2.5 commit: `ccd81054de9859faeb19b773fff0e2e1ae9e959e`

## Highlights

- Gradio Web UI with browser upload, local paths, and Google Drive input.
- Demucs dialogue/background separation and AST-based singing-vocal protection.
- Faster-Whisper, FireRedASR2/2S, Qwen3-ASR-1.7B + ForcedAligner, and Fun-ASR-Nano.
- Text/file hotwords and deterministic `wrong=>correct` substitutions.
- HY-MT2-7B/1.8B, NLLB, Chat Completions-compatible APIs, and manual translations.
- Editable JSON/SRT/VTT/ASS/SSA timelines; locked human translations survive resumed runs.
- Per-speaker timbre references, per-segment source emotion, and QwenEmotion text control.
- Natural-speed-first timing with bounded silence borrowing and exact final alignment.
- Resumable per-segment caches, isolated dubbed audio, subtitles, video, and QC reports.
- Unified Hugging Face caching, disabled Xet reconstruction, and HY-MT2 disk preflight checks.

## Run on Google Colab

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jasonBrom/IndexTTS-DubFlow/blob/main/notebooks/IndexTTS_DubFlow_Colab_R7.ipynb)

- [Visible-log Notebook](notebooks/IndexTTS_DubFlow_Colab_R7.ipynb)
- [Compact Notebook](notebooks/IndexTTS_DubFlow_Colab_R7_Compact.ipynb)

The Notebook embeds a source snapshot matching this release, so the Colab runtime does not need to clone this repository. L4, A10, and A100 are recommended. T4 falls back from BF16 to FP32 and is slower with less memory headroom.

## Platform support

| Platform | Status | Notes |
| --- | --- | --- |
| Linux x86_64 + NVIDIA | Full | Ubuntu 22.04/24.04, CUDA 12.8+, Python 3.11 recommended |
| Windows 10/11 + NVIDIA | Full | Native PowerShell installer; WSL2 recommended for complex dependency stacks |
| Windows WSL2 + NVIDIA | Full, recommended | Uses Linux commands with GPU passthrough from the Windows NVIDIA driver |
| Docker + NVIDIA Toolkit | Full | Linux or Windows WSL2 Docker with persistent runtime/output volumes |
| Google Colab | Full | L4/A10/A100 recommended; T4 works with reduced performance/headroom |
| macOS Intel/Apple Silicon | Limited | Web, tests, and some CPU stages work; the full upstream TTS path is CUDA-oriented |
| AMD/Intel GPU | Not guaranteed | The complete path targets CUDA; CPU fallback is impractical for long-form production |

See the [installation and platform matrix](docs/INSTALLATION.md) for tested boundaries.

## Quick start

### Linux / WSL2

```bash
git clone https://github.com/jasonBrom/IndexTTS-DubFlow.git
cd IndexTTS-DubFlow
chmod +x scripts/*.sh
./scripts/bootstrap.sh --model-source modelscope
./scripts/run.sh --server-name 127.0.0.1 --server-port 7860
```

### Windows PowerShell

```powershell
git clone https://github.com/jasonBrom/IndexTTS-DubFlow.git
Set-Location IndexTTS-DubFlow
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1 --model-source modelscope
.\scripts\run.ps1 --server-name 127.0.0.1 --server-port 7860
```

### Docker

```bash
GRADIO_AUTH='user:strong-password' docker compose up --build
```

Reserve at least 25 GiB for the base runtime. Qwen3-ASR and HY-MT2 need additional storage. HY-MT2 4-bit loading reduces VRAM use, not download size.

## Development

```bash
python -m pip install -e '.[test]'
python -m compileall -q app.py original_dubber scripts tests
pytest -q
ruff check .
python scripts/setup_runtime.py --dry-run
python scripts/build_artifacts.py
python scripts/build_artifacts.py --check
```

Lightweight tests do not download models. A full GPU smoke test requires installed IndexTTS 2.5 weights and an NVIDIA CUDA environment.

## License

The original orchestration code in this repository is released under the [MIT License](LICENSE). IndexTTS and every optional model retain their respective upstream terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
