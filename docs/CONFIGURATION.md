# 配置说明

## 路径环境变量

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `DUBBER_RUNTIME_ROOT` | 仓库 `.runtime` | 所有运行时的根目录 |
| `DUBBER_OUTPUT_ROOT` | `~/Videos/IndexTTS_Dubber_Outputs` | 项目输出目录 |
| `INDEXTTS_DIR` | `<runtime>/index-tts` | 官方 IndexTTS 源码与主虚拟环境 |
| `INDEXTTS_MODEL_DIR` | `<index>/checkpoints` | IndexTTS 2.5 模型 |
| `INDEXTTS_CONFIG` | `<model>/config.yaml` | 规范化后的 2.5 配置 |
| `ASR_RUNTIME_ROOT` | `<runtime>/asr-runtimes` | Qwen/FireRed/FunASR 隔离环境与权重 |
| `TRANSLATION_RUNTIME_ROOT` | `<runtime>/translation-runtimes` | HY-MT2 隔离环境 |
| `HF_HOME` | `<runtime>/cache/huggingface` | Hugging Face 单一缓存 |
| `HYMT_HF_HOME` | `HF_HOME` | 可单独把 HY-MT2 放到大容量磁盘 |

Colab Notebook 会按 `/content` 和 Google Drive 规则显式设置这些变量。

## Web 安全

默认只建议监听 `127.0.0.1`。监听 `0.0.0.0`、创建 `--share` 链接或通过反向代理公开时，应设置：

```text
GRADIO_AUTH=username:strong-password
```

不要把 HF Token、翻译 API 密钥或登录密码写进 Git 仓库、截图或 QC 报告。应用生成公开配置快照时会遮盖 API 密钥与 HF Token。

## 模型与磁盘

- `MODEL_SOURCE=modelscope`：主模型从 ModelScope 下载；
- `MODEL_SOURCE=huggingface`：主模型从 Hugging Face 下载；
- 所有 Hugging Face 子进程继承统一缓存；
- 默认 `HF_HUB_DISABLE_XET=1`，避免 Xet 临时重建副本造成峰值磁盘占用；
- HY-MT2-7B 下载体积不会因 bitsandbytes 4-bit 加载而减少；
- 在 Web 选择 `tencent/Hy-MT2-1.8B` 可显著降低磁盘和显存压力。

## 推荐组合

### 质量优先

- ASR：Qwen3-ASR-1.7B + ForcedAligner；
- 翻译：HY-MT2-7B；
- 情感：逐句原片情感音频；
- 分轨和唱歌保护：开启；
- GPU：L4/A10/A100/RTX 3090 及以上。

### 低磁盘

- ASR：Whisper large-v3-turbo 或 Qwen3-ASR（已缓存时保留）；
- 翻译：HY-MT2-1.8B、NLLB 或 API；
- `HYMT_HF_HOME` 指向独立大容量盘；
- 不要删除已完成的 Qwen ASR/ForcedAligner 模型分片。

## 任务与缓存

项目名、视频指纹和目标语言决定任务目录。逐句 TTS 缓存键还包含译文、时间窗、说话人、参考音频指纹、情感模式和随机种子，因此人工修改某句后只重做受影响片段。

