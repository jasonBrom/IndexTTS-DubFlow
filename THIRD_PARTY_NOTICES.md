# Third-party notices

The optional singing detector uses `MIT/ast-finetuned-audioset-10-10-0.4593`
under the BSD 3-Clause license. Its model card and license remain authoritative.

This project downloads and patches IndexTTS 2.5 at runtime. IndexTTS code, model weights, model outputs, and derivative works are governed by the upstream **bilibili Model Use License Agreement**, not this project's MIT license:

https://github.com/index-tts/index-tts/blob/main/LICENSE

Required upstream statement:

> 该衍生品对原模型所作的任何改动与原模型原始权利人无关，原始权利人对该衍生品不背书、不担保、不承担责任。

The setup keeps the upstream repository's copyright notices and license file in the cloned IndexTTS directory. Users must independently comply with the licenses and terms of the translation, ASR, separation, diarization, media, and input-content components they select.

Optional ASR backends are downloaded only when selected. Their authoritative code/model pages and license files remain controlling:

- FireRedASR2 / FireRedASR2S: https://github.com/FireRedTeam/FireRedASR2S
- Qwen3-ASR / Qwen3-ForcedAligner: https://github.com/QwenLM/Qwen3-ASR
- Fun-ASR-Nano / FunASR: https://github.com/modelscope/FunASR
- HY-MT2: https://huggingface.co/tencent/Hy-MT2-7B
- NLLB-200: https://huggingface.co/facebook/nllb-200-distilled-600M
- Demucs: https://github.com/facebookresearch/demucs
- Pyannote Audio: https://github.com/pyannote/pyannote-audio

Isolation of their Python dependencies does not change the upstream model licenses or usage obligations.

Optional Index-Translate and Index-Homura services use model-specific upstream terms.
The client follows the prompt and request protocols documented by the Index team;
model weights and upstream inference implementations are not bundled.
Authoritative code, model links and licenses: https://github.com/bilibili/Index-Translate
