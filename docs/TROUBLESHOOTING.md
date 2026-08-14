# 故障排查

## `No space left on device` / `os error 28`

这是磁盘问题，不是显存问题。优先：

1. 停止 Web 与仍在下载的子进程；
2. 删除 `HF_HOME/xet`、uv/pip 缓存和 `.incomplete` 临时文件；
3. 保留 Qwen3-ASR、ForcedAligner 和 HY-MT2 已完成的 Hub blobs；
4. 把 `HYMT_HF_HOME` 或整个 `DUBBER_RUNTIME_ROOT` 指向大容量磁盘；
5. 改用 HY-MT2-1.8B、NLLB 或 API。

不要在下载进程仍运行时清理缓存。

## Windows 找不到 `.venv/bin/python`

R7 已改用 `.venv/Scripts/python.exe`。若仍出现旧路径，说明运行的是 R6 或更旧源码；检查 Web 标题中的构建号应为 `2026.08.11-r7-ccd8105`。

## 找不到 Git / FFmpeg / FFprobe

安装后重开终端，并执行：

```powershell
Get-Command git
Get-Command ffmpeg
Get-Command ffprobe
```

路径存在但命令不可见时，将对应 `bin` 目录加入系统 PATH。

## CUDA 不可用

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Windows 原生优先检查 NVIDIA 驱动和 CUDA 12.8+；WSL2 只安装 Windows 主机驱动，然后在 WSL 内确认 `nvidia-smi`。

## T4 提示 BF16

T4 不支持 BF16，应用会自动使用 FP32。它会增加显存与时间开销，不影响模型文件正确性。优先 L4/A10/A100；T4 上让 ASR、翻译和 TTS 串行，不要并发加载。

## Web 启动了，但点击分析报错

Web 启动不会加载大模型。点击分析后才会创建可选 ASR/翻译运行时并下载权重。查看后台完整日志，而不是只看 Gradio 的最后一行包装错误。

## Windows 上 Qwen3-ASR 报 `transformers.masking_utils`

这是旧版 R7 在 Windows 上把 IndexTTS 父环境排在 Qwen 子环境之前造成的依赖串用。更新到包含 `.ready-v3` 运行时迁移的版本后再次选择 Qwen3-ASR，程序会自动移动父环境桥接文件、关闭 `system-site-packages`，并在写入就绪标记前验证 `transformers==4.57.6` 与 Qwen3-ASR 导入。

可用下面的命令确认实际加载位置：

```powershell
$Py = ".\.runtime\asr-runtimes\qwen3-asr-1.7b\.venv\Scripts\python.exe"
& $Py -c "import transformers; print(transformers.__version__); print(transformers.__file__)"
```

正确结果应为 `4.57.6`，且路径位于 `asr-runtimes\qwen3-asr-1.7b\.venv\Lib\site-packages`，而不是 `index-tts\.venv`。

## 配置自检失败

运行：

```bash
<运行时Python> scripts/normalize_indextts25_config.py <模型目录>/config.yaml
<运行时Python> scripts/doctor.py
```

若报告真正的 2.0 架构、补丁无法应用或固定提交不一致，不要强行绕过；清理被手工修改的 IndexTTS 源码目录后重新安装。

## `alimiter` 不支持 `latency`

项目会自动使用 5 ms 手动延迟补偿，并保持最终长度。若仍失败，升级 FFmpeg，并确认执行的是 PATH 中预期版本。

## macOS / 非 NVIDIA GPU 很慢

这是当前上游 CUDA 依赖边界。Web、字幕、NLLB、部分 ASR 可在 CPU 跑，但完整视频 TTS 不保证具有可接受速度。不要把安装成功等同于完整 GPU 推理已受支持。
