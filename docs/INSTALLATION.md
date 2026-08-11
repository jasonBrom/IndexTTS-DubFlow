# 安装与平台说明

## 1. 支持级别

| 环境 | 安装入口 | GPU | 完整译制 |
| --- | --- | --- | --- |
| Ubuntu 22.04/24.04 x86_64 | `scripts/bootstrap.sh` | NVIDIA CUDA | 是 |
| Windows 10/11 x64 原生 | `scripts/bootstrap.ps1` | NVIDIA CUDA | 是，依赖冲突时改用 WSL2 |
| Windows WSL2 Ubuntu | `scripts/bootstrap.sh` | NVIDIA CUDA 透传 | 是，Windows 推荐路径 |
| Google Colab | Release Notebook | Colab NVIDIA GPU | 是 |
| Docker + NVIDIA Container Toolkit | `docker compose` | NVIDIA CUDA | 是 |
| macOS x86_64/arm64 | `scripts/bootstrap.sh` | CPU/MPS 取决于上游 | 仅有限支持，不建议完整出片 |

上游 IndexTTS 当前明确给出 Linux/Windows 安装说明，并建议 CUDA Toolkit 12.8 或更新版本。Windows 上不安装 DeepSpeed；本项目只同步官方 `webui` extra。

## 2. 前置条件

- Python 3.11 x64；
- Git；
- FFmpeg 与 FFprobe，且位于 `PATH`；
- NVIDIA 驱动；Windows/Linux GPU 环境建议 CUDA 12.8+；
- 基础安装至少 25 GiB 可用空间；
- 下载 Qwen3-ASR + ForcedAligner、HY-MT2-7B 时需要更多空间。

验证：

```bash
python --version
git --version
ffmpeg -version
ffprobe -version
nvidia-smi
```

Windows 可用：

```powershell
winget install Python.Python.3.11
winget install Git.Git
winget install Gyan.FFmpeg
```

安装后必须重开 PowerShell，使新 PATH 生效。

## 3. Linux / WSL2

```bash
chmod +x scripts/*.sh
./scripts/bootstrap.sh --model-source modelscope
./scripts/run.sh
```

Hugging Face 下载源：

```bash
./scripts/bootstrap.sh --model-source huggingface
```

安装到其他磁盘：

```bash
./scripts/bootstrap.sh --runtime-root /mnt/models/IndexTTS25-Runtime
export DUBBER_RUNTIME_ROOT=/mnt/models/IndexTTS25-Runtime
./scripts/run.sh
```

WSL2 只安装 Windows NVIDIA 显卡驱动；不要在 WSL 内重复安装 Windows 驱动。进入 WSL 后先确认 `nvidia-smi` 可见 GPU。

## 4. Windows 原生

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1 --model-source modelscope
.\scripts\run.ps1
```

安装到 D 盘：

```powershell
.\scripts\bootstrap.ps1 --runtime-root D:\IndexTTS25-Runtime
$env:DUBBER_RUNTIME_ROOT = "D:\IndexTTS25-Runtime"
.\scripts\run.ps1
```

如 Git、FFmpeg、CUDA 扩展或某个 ASR 的依赖在原生 Windows 上失败，优先使用 WSL2；项目输出可直接放在 `/mnt/d/...`，但模型和虚拟环境建议放在 WSL 的 Linux 文件系统以避免大量小文件性能下降。

## 5. macOS

```bash
brew install python@3.11 git ffmpeg
PYTHON_BIN=python3.11 ./scripts/bootstrap.sh --skip-models
```

`--skip-models` 适合先验证 Web、配置和轻量测试。完整 IndexTTS 2.5 推理链路以上游 CUDA 方案为主；即使环境能安装，CPU 出片也可能非常慢，部分 CUDA 专用依赖不可用。

## 6. Docker

Docker 镜像基于 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`。主机需要 NVIDIA 驱动、Docker Compose 与 NVIDIA Container Toolkit。

```bash
GRADIO_AUTH='user:strong-password' docker compose up --build
```

运行环境保存在命名卷 `indextts25-runtime`，输出映射到仓库 `outputs/`。删除容器不会删除模型；删除命名卷会删除全部已下载运行时。

## 7. 可选安装参数

```text
--runtime-root PATH
--index-dir PATH
--model-dir PATH
--model-source modelscope|huggingface
--with-demucs / --no-with-demucs
--with-singing-detector / --no-with-singing-detector
--with-diarization / --no-with-diarization
--clean-cache / --no-clean-cache
--skip-models
--min-free-gib 25
--ignore-space-check
--dry-run
```

`--ignore-space-check` 只关闭安装前保护，不会减少实际空间需求。

## 8. 自检

安装器会自动运行：

```bash
<IndexTTS虚拟环境Python> scripts/doctor.py
```

自检要求：固定提交、补丁、2.5 配置结构、模型文件、辅助模型和所选 Python 模块全部存在。没有 CUDA 只给警告；完整视频性能仍不受保证。
