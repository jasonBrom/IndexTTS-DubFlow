# Index-Translate 接入与研究说明

核对时间：2026-10-05（北京时间）。本次基于 DubFlow 主分支 `c679dfb`，保留 Windows Qwen3-ASR 隔离修复。

## 模型选型

Index 在 2026-09-30 发布的是基于 Qwen3.5 的翻译模型家族，与 IndexTTS 2.5 的语音合成职责不同。

| 模型 | 官方用途 | 本次接入 |
| --- | --- | --- |
| Index-Translate 2B / 9B / 35B-A3B-preview | 150 语种文本翻译，术语和风格约束 | 接入翻译阶段，支持自建兼容服务；另有官方公网 35B API 预设 |
| Index-Homura 2B / 9B | 指定目标音节数的翻译 | 接入逐句音节预算，译文交给现有 IndexTTS 配音 |
| Index-Echo S2TT / S2ST | 语音转译文或直接生成译配语音 | 已研究，未接入此次文本翻译后端 |
| Index-NativeLong（仓库 ID 为 Index-Nailong） | 整篇长文翻译 | 未接入逐句视频流程 |

选择 Translate/Homura 是为了继续使用现有的 ASR、人工字幕锁定、说话人参考音频、情感控制、TTS 时间规划和背景回混。Echo S2ST 的官方管线包含 ST LM、Hidden2CV mapper 和 CosyVoice3，需要独立语音后端，不是给 Chat Completions 换一个模型名即可接入。当前发布脚本的方向为中→英/西/日、英→中/西/日，建议单句不超过 30 秒。NativeLong 的固定模板支持中英和中日双向，整篇译文也不能直接保证逐行字幕对应。

文本翻译覆盖 150 语种不代表 DubFlow 的配音输出也支持 150 语种。本次保留 IndexTTS 2.5 原有的中、英、日、西、阿五种目标语言。

## 最快使用：官方公网 API

打开 R8 的 Web 面板，在「高级设置 → 翻译方式」选择 **Index-Translate 官方公网 API**。无需填写密钥、模型地址，也不下载翻译权重。

- 官方地址：`https://index-translate.bilibili.com/v1`
- 公网模型名：`Index-Translate-35B-A3B`（与 Hugging Face 仓库 ID 不同）
- 官方于 2026-10-04 公布免费调用；服务可用性、额度与限流可能调整。
- 会发送当前原文、配置的前后文、前文译文、术语及风格；此接口不上传音视频。
- 需要本地处理文本时，选择自建服务。原来的默认 HY-MT2 选择保留，不会自动切换到公网。

不加载任何 ASR/TTS 权重，单独检查连接：

```bash
python scripts/check_index_translation.py --backend public
```

本次对上述真实服务完成了一次请求（中文→英文，关闭思考），输出为 `Hello, world. The weather is nice today.`。这是连通性验证，不是翻译质量基准测试。

## 自建服务：Translate / Homura

本次实现兼容 API 客户端；选择本地模式不会自动安装 vLLM 或加载权重。推荐在独立 Linux/WSL2 CUDA 环境启动服务，避免把 Qwen3.5 的依赖装入固定版本的 IndexTTS 环境。官方文本推理文档核对过 vLLM 0.29。

以下以已有 Python 3.12 和 uv 的 Linux 环境为例；命令启动在前台，另开终端运行 DubFlow。不要把新虚拟环境放在仓库内，以免打入源码包。

```bash
uv venv --python 3.12 ../index-translate-server-env
uv pip install --python ../index-translate-server-env/bin/python 'vllm==0.29.0'
../index-translate-server-env/bin/vllm serve IndexTeam/Index-Translate-2B \
  --host 127.0.0.1 --port 8000 --max-model-len 4096
```

界面选「Index-Translate 本地/自建服务」，地址 `http://127.0.0.1:8000/v1`，模型 `IndexTeam/Index-Translate-2B`，无鉴权可留空密钥。同样可以填写 9B、35B preview、官方量化版本或服务端自定义名称，但必须与该服务实际加载的模型一致。

Homura 请停止上述服务，再用对应权重启动：

```bash
../index-translate-server-env/bin/vllm serve IndexTeam/Index-Homura-2B \
  --host 127.0.0.1 --port 8000 --max-model-len 4096
python scripts/check_index_translation.py '我们去看电影吧' \
  --backend homura --seconds 2 --syllables-per-second 4 --glossary '电影=cinema'
```

界面选择「Index-Homura 音节控制（自建服务）」，模型应为 `IndexTeam/Index-Homura-2B`。2B/9B 可选择，量化仓库或别名可手工输入。官方已发布 GGUF、FP8 等版本；GGUF 可由支持对应架构的 llama.cpp 提供兼容服务，客户端需填写该服务暴露的模型名称。本次未实测本地量化推理。Echo 的 GGUF 只包含文本主干，不能当作完整语音模型。

Colab 内的 `127.0.0.1` 指 Colab 运行时。电脑上的服务器必须通过可访问的服务地址连接。自建服务与 TTS 同用一张卡时，服务常驻仍会占显存，DubFlow 不负责自动卸载另一个进程中的模型；低显存场景优先用公网接口或独立翻译机器。

## Homura 的长度控制

本次预算是 `max(1, floor(原句时长 × 每秒音节数 + 0.5))`。默认 4.5 音节/秒是本项目的可调经验起点，不是官方标准；各目标语言、角色和语气应分别校准。例如 3 秒、4.5 音节/秒产生 14 音节目标。

Homura 控制的是译文音节数，不是 WAV 秒数，也不是字数或 token 数。模型也不保证每次满足音节约束；当前未增加多语言音节计数器和自动纠错回环。过小预算可能压缩语义，应检查译文，必要时提高语速参数或改用普通 Translate。配音阶段仍优先自然语速、利用句间空白并执行现有时长适配。

## 接入细节

- 单条 user 消息、`chat_template_kwargs.enable_thinking=false`；Translate 温度 0，Homura 温度 0.3，与官方客户端默认策略一致。
- Translate 使用官方 instTrans 格式；Homura 使用指定音节数模板。项目另加可配置的上下文和风格信息；关闭上下文且清空风格可接近官方最小 Homura 模板。
- 术语表沿用界面每行 `原词=译词`，也接受冒号、箭头或 JSON 字符串字典；不会把术语表输入解释为文件路径。
- 默认最大输出 1024 token，界面可调 128～8192。服务端总上下文必须容纳提示词和输出；长字幕或大量上下文可调低上下文句数或增加服务窗口。
- 明确报错处理空译文、无效 JSON、思考残留、截断结果；429 和暂时性服务器/连接错误最多重试两次，不回退为原文，不自动改用其他服务。
- 每次只填未锁定的空译文；保留时间戳、说话人和人工修改。Index 参数参与分析缓存指纹，密钥在公开配置中遮盖；公网模式不会带上自建服务密钥。

## 验证边界

完成官方公网真实请求、自动化 HTTP 集成/错误处理测试和原有回归测试；本机无 CUDA，未验证 Homura/Translate 本地权重推理、显存用量或端到端视频配音。新增入口属于 API 服务接入，并非内置本地模型加载器。

## 官方资料

- [官方中文总览、发布日期及公网 API](https://github.com/bilibili/Index-Translate/blob/main/README_zh.md)
- [文本模型部署与模板](https://github.com/bilibili/Index-Translate/blob/main/inference/llm/README.md)
- [Translate 官方客户端](https://github.com/bilibili/Index-Translate/blob/main/inference/llm/translate.py)
- [Homura 官方客户端](https://github.com/bilibili/Index-Translate/blob/main/inference/llm/syllable_translate.py)
- [Echo S2ST](https://github.com/bilibili/Index-Translate/blob/main/inference/echo-s2st/README_zh.md)
- [Echo S2TT](https://github.com/bilibili/Index-Translate/blob/main/inference/echo-s2tt/README_zh.md)
