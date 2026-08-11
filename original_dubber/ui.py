from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr

from .asr import parse_hotword_spec
from .media import scan_drive_videos
from .models import DubConfig, Segment
from .pipeline import OriginalVoiceDubber
from .platforms import output_root as platform_output_root
from .platforms import runtime_paths
from .subtitles import import_reviewed_timeline
from .utils import file_fingerprint, safe_name, write_srt

BUILD_VERSION = "2026.08.11-r7-ccd8105"
TIMELINE_HEADERS = [
    "启用", "开始秒", "原结束秒", "说话人", "原文", "译文", "锁定人工译文",
    "情感描述", "原时长", "实际结束秒", "自然时长", "速度倍率", "状态",
]


def _default_output_root() -> str:
    return str(platform_output_root())


def _optional_text(value: Any) -> str:
    """Normalize nullable Gradio text values before building a config."""
    return "" if value is None else str(value)


def timeline_rows(segments: list[Segment]) -> list[list[Any]]:
    return [
        [
            item.enabled,
            round(item.start, 3),
            round(item.end, 3),
            item.speaker,
            item.source_text,
            item.target_text,
            item.translation_locked,
            item.emotion_text,
            round(item.duration, 3),
            round(item.render_end, 3) if item.render_end is not None else None,
            round(item.natural_duration, 3) if item.natural_duration is not None else None,
            round(item.duration_factor, 3) if item.duration_factor is not None else None,
            item.warning,
        ]
        for item in segments
    ]


def rows_to_segments(data: Any, state: Any) -> list[Segment]:
    if hasattr(data, "values"):
        rows = data.values.tolist()
    else:
        rows = data or []
    originals = state.get("segments", []) if isinstance(state, dict) else state
    if len(rows) != len(originals):
        raise ValueError("时间轴行数发生变化。当前版本允许编辑，但暂不允许直接增删行。")
    segments: list[Segment] = []
    for index, (row, original) in enumerate(zip(rows, originals, strict=False)):
        if len(row) < 8:
            raise ValueError(f"第 {index + 1} 行缺少字段")
        segment = Segment.from_dict(original)
        segment.id = index
        segment.enabled = bool(row[0])
        segment.start = float(row[1])
        segment.end = float(row[2])
        segment.speaker = str(row[3]).strip() or "SPEAKER_00"
        if row[4] is None or row[5] is None:
            raise ValueError(f"第 {index + 1} 行原文或译文为空")
        segment.source_text = str(row[4]).strip()
        target_text = str(row[5]).strip()
        manually_changed = target_text != segment.target_text.strip()
        segment.target_text = target_text
        segment.translation_locked = bool(row[6]) or (manually_changed and bool(target_text))
        segment.emotion_text = str(row[7] or "").strip()
        if (
            manually_changed
            or segment.start != float(original.get("start", segment.start))
            or segment.end != float(original.get("end", segment.end))
        ):
            segment.generated_path = ""
            segment.render_start = None
            segment.render_end = None
            segment.max_render_end = None
        segment.validate()
        segments.append(segment)
    return segments


def _resolve_source(mode: str, upload: Any, local_path: str, drive_path: str) -> str:
    if mode == "上传视频":
        if upload is None:
            raise ValueError("请上传视频")
        if isinstance(upload, dict):
            return str(upload.get("path") or upload.get("name"))
        return str(upload)
    if mode == "Google Drive":
        if not drive_path:
            raise ValueError("请选择 Google Drive 视频")
        return drive_path
    if not local_path.strip():
        raise ValueError("请输入 Colab/本地视频路径")
    return str(Path(local_path).expanduser())


def _upload_path(upload: Any) -> str:
    if upload is None:
        return ""
    if isinstance(upload, dict):
        return str(upload.get("path") or upload.get("name") or "")
    return str(upload)


def _read_uploaded_text(upload: Any) -> str:
    path = _upload_path(upload)
    if not path:
        return ""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "gb18030", "big5"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _make_config(
    source_mode: str,
    upload: Any,
    local_path: str,
    drive_path: str,
    project_name: str,
    output_root: str,
    source_language: str,
    target_language: str,
    asr_model: str,
    input_subtitle: Any,
    input_subtitle_mode: str,
    hotword_text: str,
    hotword_file: Any,
    translation_backend: str,
    nllb_model: str,
    hymt2_model: str,
    translation_context_segments: int,
    translation_style: str,
    translation_glossary: str,
    hymt2_quantization: str,
    llm_api_base: str,
    llm_api_key: str,
    llm_model: str,
    separate_background: bool,
    protect_singing_vocals: bool,
    diarization: bool,
    hf_token: str,
    emotion_mode: str,
    emotion_strength: float,
    subtitle_mode: str,
    original_audio_volume: float,
    use_bf16: bool,
    use_cuda_kernel: bool,
    authorized: bool,
) -> DubConfig:
    if not authorized:
        raise ValueError("必须确认已获得视频、声音和翻译所需授权，才能开始处理。")

    # Gradio may return None for every unfilled optional Textbox.  Normalize all
    # text-like inputs at the UI boundary so downstream code can safely strip,
    # split and persist them without having to understand Gradio's value model.
    source_mode = _optional_text(source_mode) or "上传视频"
    local_path = _optional_text(local_path)
    drive_path = _optional_text(drive_path)
    project_name = _optional_text(project_name)
    output_root = _optional_text(output_root).strip() or _default_output_root()
    source_language = _optional_text(source_language) or "auto"
    target_language = _optional_text(target_language) or "ZH"
    asr_model = _optional_text(asr_model) or "whisper:large-v3-turbo"
    input_subtitle_mode = (
        _optional_text(input_subtitle_mode) or "不上传字幕，使用 ASR"
    )
    hotword_text = _optional_text(hotword_text)
    translation_backend = (
        _optional_text(translation_backend) or "HY-MT2-7B 本地（推荐）"
    )
    nllb_model = (
        _optional_text(nllb_model) or "facebook/nllb-200-distilled-600M"
    )
    hymt2_model = _optional_text(hymt2_model) or os.environ.get(
        "HYMT_DEFAULT_MODEL", "tencent/Hy-MT2-7B"
    )
    translation_style = _optional_text(translation_style)
    translation_glossary = _optional_text(translation_glossary)
    hymt2_quantization = _optional_text(hymt2_quantization) or "auto"
    llm_api_base = _optional_text(llm_api_base)
    llm_api_key = _optional_text(llm_api_key)
    llm_model = _optional_text(llm_model)
    hf_token = _optional_text(hf_token)
    emotion_mode = _optional_text(emotion_mode) or "segment"
    subtitle_mode = _optional_text(subtitle_mode) or "soft"

    source = _resolve_source(source_mode, upload, local_path, drive_path)
    automatic_project_name = (
        f"dub_{safe_name(Path(source).stem, 'video')}_"
        f"{file_fingerprint(source)[:8]}_{target_language}"
    )
    backend_map = {
        "HY-MT2-7B 本地（推荐）": "hymt2",
        "NLLB-200 离线": "nllb",
        "兼容 Chat Completions API": "llm",
        "不翻译/手工编辑": "none",
    }
    subtitle_mode_map = {
        "不上传字幕，使用 ASR": "none",
        "作为原文字幕（需要翻译）": "source",
        "作为译文字幕（直接锁定）": "translated",
    }
    hotwords, replacements = parse_hotword_spec(hotword_text, _read_uploaded_text(hotword_file))
    return DubConfig(
        source_video=source,
        output_root=output_root,
        project_name=project_name.strip() or automatic_project_name,
        source_language=source_language,
        target_language=target_language,
        asr_model=asr_model,
        asr_hotwords=hotwords,
        asr_replacements=replacements,
        asr_runtime_root=os.environ.get("ASR_RUNTIME_ROOT", str(runtime_paths().asr)),
        model_source=os.environ.get("MODEL_SOURCE", "modelscope"),
        input_subtitle_path=_upload_path(input_subtitle),
        input_subtitle_mode=subtitle_mode_map[input_subtitle_mode],
        translation_backend=backend_map[translation_backend],
        translation_model=(hymt2_model if backend_map[translation_backend] == "hymt2" else nllb_model),
        translation_runtime_root=os.environ.get(
            "TRANSLATION_RUNTIME_ROOT", str(runtime_paths().translation)
        ),
        translation_context_segments=int(
            3 if translation_context_segments is None else translation_context_segments
        ),
        translation_style=translation_style.strip(),
        translation_glossary=translation_glossary.strip(),
        hymt2_quantization=hymt2_quantization,
        llm_api_base=llm_api_base,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        separate_background=separate_background,
        protect_singing_vocals=protect_singing_vocals,
        diarization=diarization,
        hf_token=hf_token,
        emotion_mode=emotion_mode,
        emotion_strength=float(0.72 if emotion_strength is None else emotion_strength),
        enable_qwen_emotion=True,
        subtitle_mode=subtitle_mode,
        original_audio_volume=float(
            0.18 if original_audio_volume is None else original_audio_volume
        ),
        use_bf16=use_bf16,
        use_cuda_kernel=use_cuda_kernel,
        indextts_dir=os.environ.get("INDEXTTS_DIR", str(runtime_paths().indextts)),
        model_dir=os.environ.get("INDEXTTS_MODEL_DIR", str(runtime_paths().checkpoints)),
        config_path=os.environ.get(
            "INDEXTTS_CONFIG", str(runtime_paths().checkpoints / "config.yaml")
        ),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="IndexTTS-DubFlow", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# IndexTTS-DubFlow\n"
            f"**Web 构建：{BUILD_VERSION}**。上传视频或选择 Google Drive 文件，自动完成分轨、"
            "识别、上下文翻译、原音色/情感配音、自然语速时间规划与视频封装。\n\n"
            "> 已跟进 IndexTTS 2.5 正式合并到官方 `main` 的提交 `ccd8105`：包含 KV cache、"
            "长文本切分、发音标注保护及新模型目录配置。本项目仅为无法自然容纳的句子保留"
            "S2M Mel 帧级目标时长补丁，不冒充论文中尚未开放的 T2S token-count 接口。"
        )
        timeline_state = gr.State({})

        with gr.Tab("1. 新建与分析"):
            with gr.Row():
                source_mode = gr.Radio(
                    ["上传视频", "输入路径", "Google Drive"], value="上传视频", label="视频来源"
                )
                project_name = gr.Textbox(
                    label="项目名称",
                    value="",
                    placeholder="留空则每次任务自动生成时间戳名称",
                )
                output_root = gr.Textbox(label="输出目录", value=_default_output_root())
            with gr.Row():
                video_upload = gr.Video(label="上传视频", sources=["upload"], format=None)
                local_path = gr.Textbox(
                    label="Colab/本地路径",
                    placeholder=str(Path.home() / "Videos" / "input.mp4"),
                )
            with gr.Row():
                drive_path = gr.Dropdown(
                    choices=[], label="Google Drive 视频", allow_custom_value=True
                )
                refresh_drive = gr.Button("刷新云盘文件")
            with gr.Row():
                source_language = gr.Dropdown(
                    choices=[
                        ("自动识别", "auto"), ("中文", "zh"), ("英语", "en"),
                        ("日语", "ja"), ("西班牙语", "es"), ("阿拉伯语", "ar"),
                        ("韩语", "ko"), ("法语", "fr"), ("德语", "de"),
                        ("俄语", "ru"), ("葡萄牙语", "pt"), ("越南语", "vi"),
                    ],
                    value="auto",
                    label="源语言",
                )
                target_language = gr.Dropdown(
                    choices=[("中文", "ZH"), ("English", "EN"), ("日本語", "JA"), ("Español", "ES"), ("العربية", "AR")],
                    value="ZH",
                    label="目标语言（IndexTTS 2.5）",
                )
                asr_model = gr.Dropdown(
                    [
                        ("Whisper large-v3-turbo", "whisper:large-v3-turbo"),
                        ("Whisper large-v3", "whisper:large-v3"),
                        ("Whisper medium", "whisper:medium"),
                        ("Whisper small", "whisper:small"),
                        ("FireRedASR2-AED", "fireredasr2-aed"),
                        ("FireRedASR2S 完整系统", "fireredasr2s"),
                        ("Qwen3-ASR-1.7B + ForcedAligner", "qwen3-asr-1.7b"),
                        ("Fun-ASR-Nano-2512", "fun-asr-nano-2512"),
                    ],
                    value="whisper:large-v3-turbo",
                    label="ASR 模型（Whisper / FireRedASR2 / Qwen3-ASR / Fun-ASR）",
                )
            gr.Markdown(
                "当前可见 ASR：Whisper large-v3-turbo / large-v3 / medium / small、"
                "FireRedASR2-AED / FireRedASR2S、Qwen3-ASR-1.7B + ForcedAligner、"
                "Fun-ASR-Nano-2512。非 Whisper 模型首次选择时会按需安装隔离运行环境。"
            )
            with gr.Row():
                hotword_text = gr.Textbox(
                    label="ASR 热词库（直接填写）",
                    lines=5,
                    placeholder="每行一个词；也支持 错词=>正确词\n例如：IndexTTS\n因得克斯=>IndexTTS",
                )
                hotword_file = gr.File(
                    label="上传热词库 TXT（可选）", file_types=[".txt"], type="filepath"
                )
            gr.Markdown(
                "普通热词会传给支持该能力的 Whisper、Qwen3-ASR、Fun-ASR-Nano；"
                "`错词=>正确词` 会对所有 ASR 做可复现的结果纠正。"
            )
            with gr.Row():
                input_subtitle = gr.File(
                    label="手动上传字幕（SRT/VTT/ASS/SSA，可选）",
                    file_types=[".srt", ".vtt", ".ass", ".ssa"],
                    type="filepath",
                )
                input_subtitle_mode = gr.Radio(
                    ["不上传字幕，使用 ASR", "作为原文字幕（需要翻译）", "作为译文字幕（直接锁定）"],
                    value="不上传字幕，使用 ASR",
                    label="字幕用途",
                )
            gr.Markdown(
                "上传字幕后会跳过 ASR，并保持字幕原时间轴。作为原文时必须明确选择源语言；"
                "作为译文时不会再次翻译，所有行自动锁定。"
            )
            authorized = gr.Checkbox(
                label="我确认拥有或已获得该视频、人物声音、字幕翻译及输出用途所需的合法授权",
                value=False,
            )
            with gr.Row():
                analyze_button = gr.Button("分析、识别并翻译", variant="primary")
                one_click_button = gr.Button("一键分析并生成视频", variant="primary")
            status = gr.Markdown("等待任务")
            project_path = gr.Textbox(label="任务目录", interactive=False)

        with gr.Tab("2. 时间轴编辑与生成"):
            gr.Markdown(
                "可编辑开始/结束时间、说话人标签和译文。把同一人物的片段设成相同说话人标签，"
                "系统会自动重建该人物的音色参考。检测为唱歌的行默认关闭；如需重新启用，"
                "请同时手工填写译文。手动改动译文会自动勾选“锁定人工译文”，后续分析不会覆盖；"
                "也可手动解除锁定。`情感描述` 可填“克制的悲伤、轻微颤抖”等自然语言。"
                "实际结束时间允许借用下一句之前的真实静音，但不会推迟句首或覆盖下一句。"
            )
            timeline = gr.Dataframe(
                headers=TIMELINE_HEADERS,
                datatype=[
                    "bool", "number", "number", "str", "str", "str", "bool", "str",
                    "number", "number", "number", "number", "str",
                ],
                value=[],
                row_count=(0, "fixed"),
                col_count=(len(TIMELINE_HEADERS), "fixed"),
                interactive=True,
                wrap=True,
                label="对白时间轴",
            )
            with gr.Row():
                reviewed_timeline_file = gr.File(
                    label="上传人工校对时间轴（JSON/SRT/VTT/ASS/SSA）",
                    file_types=[".json", ".srt", ".vtt", ".ass", ".ssa"],
                    type="filepath",
                )
                import_timeline_button = gr.Button("导入校对结果")
            with gr.Row():
                save_timeline_button = gr.Button("保存时间轴与人工译文")
                export_timeline_button = gr.Button("导出时间轴供人工校对")
                render_button = gr.Button("按当前时间轴生成/继续生成", variant="primary")
            result_video = gr.Video(label="译制结果")
            timeline_export_files = gr.File(label="时间轴导出", file_count="multiple")
            result_files = gr.File(label="项目文件", file_count="multiple")
            qc_report = gr.Markdown()

        with gr.Tab("3. 高级设置"):
            translation_backend = gr.Radio(
                ["HY-MT2-7B 本地（推荐）", "NLLB-200 离线", "兼容 Chat Completions API", "不翻译/手工编辑"],
                value="HY-MT2-7B 本地（推荐）",
                label="翻译方式",
            )
            with gr.Row():
                hymt2_model = gr.Dropdown(
                    choices=[
                        ("Hy-MT2-7B 高质量（原始权重 16.1 GB）", "tencent/Hy-MT2-7B"),
                        ("Hy-MT2-1.8B 节省磁盘（约 4 GB）", "tencent/Hy-MT2-1.8B"),
                    ],
                    value=os.environ.get("HYMT_DEFAULT_MODEL", "tencent/Hy-MT2-7B"),
                    allow_custom_value=True,
                    label="HY-MT2 模型",
                )
                hymt2_quantization = gr.Dropdown(
                    choices=[("自动（显存不足用 4-bit）", "auto"), ("4-bit", "4bit"), ("BF16", "bf16")],
                    value="auto",
                    label="HY-MT2 加载精度",
                )
                nllb_model = gr.Textbox(
                    value="facebook/nllb-200-distilled-600M", label="NLLB 模型（备用）"
                )
            with gr.Row():
                translation_context_segments = gr.Slider(
                    0, 8, value=3, step=1, label="翻译上下文句数（前后各取）"
                )
                translation_style = gr.Textbox(
                    value="自然、准确、符合人物身份的影视对白；保留语气、情感和口语节奏",
                    label="翻译风格",
                )
            translation_glossary = gr.Textbox(
                label="翻译术语表（每行：原词=译词）",
                lines=4,
                placeholder="IndexTTS=IndexTTS\nGPT=GPT",
            )
            gr.Markdown(
                "HY-MT2 会同时参考前文译文、后文原文、人物标签、原句时长、风格和术语表；"
                "原句时长是软约束，优先正确与自然，再由配音时间规划处理语言时长差异。"
                "T4 的 4-bit 只节省显存，7B 仍需下载完整 16.1 GB 权重；磁盘紧张时可选 1.8B。"
            )
            with gr.Row():
                llm_api_base = gr.Textbox(label="兼容 API 地址", placeholder="https://.../v1")
                llm_model = gr.Textbox(label="API 模型名")
                llm_api_key = gr.Textbox(label="API 密钥", type="password")
            with gr.Row():
                separate_background = gr.Checkbox(
                    label="Demucs 分离原对白与背景",
                    value=os.environ.get("SEPARATE_BACKGROUND_DEFAULT", "1") == "1",
                )
                protect_singing_vocals = gr.Checkbox(
                    label="保留歌曲人声（检测唱歌，只移除对白）",
                    value=os.environ.get("PROTECT_SINGING_DEFAULT", "1") == "1",
                )
                diarization = gr.Checkbox(label="Pyannote 自动说话人分离（可选安装）", value=False)
                hf_token = gr.Textbox(label="Hugging Face token", type="password")
            with gr.Row():
                emotion_mode = gr.Radio(
                    [
                        ("逐句原片情感音频（推荐）", "segment"),
                        ("自然语言情感描述（QwenEmotion）", "text"),
                        ("仅使用人物音色参考", "speaker"),
                    ],
                    value="segment",
                    label="情感模式",
                )
                emotion_strength = gr.Slider(
                    0.0, 1.0, value=0.72, step=0.01, label="情感控制强度"
                )
                subtitle_mode = gr.Radio(
                    [("内封可关闭字幕", "soft"), ("烧录字幕", "burn"), ("不封装字幕", "none")],
                    value="soft",
                    label="字幕方式",
                )
                original_audio_volume = gr.Slider(
                    0.0, 1.0, value=0.18, step=0.01,
                    label="关闭分轨时的原音轨音量",
                )
            with gr.Row():
                use_bf16 = gr.Checkbox(label="支持时使用 BF16（T4 会自动禁用）", value=True)
                use_cuda_kernel = gr.Checkbox(label="实验性 BigVGAN CUDA kernel", value=False)
            gr.Markdown(
                "情感音频与音色参考分离：当前句太短时只拼接最近的同一说话人片段，不会把相邻人物混入。"
                "在译文开头写 `[情感:极度悲伤，声音颤抖]` 可逐句覆盖情感描述；该项目标签会在朗读前删除，"
                "内容送入官方 QwenEmotion 文本控制，不会被当成对白念出。"
            )

        with gr.Tab("4. 说明与许可"):
            gr.Markdown(
                "- IndexTTS 2.5 使用自定义 **bilibili Model Use License**，不是 MIT/Apache。\n"
                "- 不得克隆未授权个人或公众人物声音，不得用于欺诈、冒充或违法用途。\n"
                "- 逐句总时长对齐不等于逐音素唇形同步；复杂重叠对白仍建议人工复核。\n"
                "- HY-MT2、NLLB 及其他依赖模型另有各自许可证；商业项目应自行逐项核对。"
            )

        common_inputs = [
            source_mode, video_upload, local_path, drive_path, project_name, output_root,
            source_language, target_language, asr_model,
            input_subtitle, input_subtitle_mode, hotword_text, hotword_file,
            translation_backend, nllb_model, hymt2_model, translation_context_segments,
            translation_style, translation_glossary, hymt2_quantization,
            llm_api_base, llm_api_key, llm_model, separate_background, protect_singing_vocals,
            diarization, hf_token,
            emotion_mode, emotion_strength, subtitle_mode, original_audio_volume,
            use_bf16, use_cuda_kernel, authorized,
        ]

        def analyze_ui(*values, progress=gr.Progress(track_tqdm=True)):
            try:
                config = _make_config(*values)
                dubber = OriginalVoiceDubber(config)
                segments = dubber.analyze(
                    progress=lambda value, text: progress(value, desc=text)
                )
                state = {
                    "segments": [item.to_dict() for item in segments],
                    "analysis_key": dubber._analysis_key(),
                    "project_root": str(dubber.paths.root),
                    "source_language": segments[0].source_language or config.source_language,
                }
                timeline_files = _export_timeline_files(dubber, segments)
                return (
                    timeline_rows(segments),
                    state,
                    f"✅ 分析完成：{len(segments)} 段。可先编辑时间轴，再生成。",
                    str(dubber.paths.root),
                    timeline_files,
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def _validated_timeline(data, state, values):
            config = _make_config(*values)
            dubber = OriginalVoiceDubber(config)
            if not isinstance(state, dict) or (
                state.get("analysis_key") != dubber._analysis_key()
                or state.get("project_root") != str(dubber.paths.root)
            ):
                raise ValueError("视频、项目或分析设置已经变化，请重新执行“分析、识别并翻译”。")
            segments = rows_to_segments(data, state)
            source_language_value = state.get("source_language") or config.source_language
            return config, dubber, segments, source_language_value

        def _export_timeline_files(dubber, segments):
            translated = write_srt(
                dubber.paths.subtitles / "translated_review.srt", segments, translated=True
            )
            source = write_srt(
                dubber.paths.subtitles / "source_reference.srt", segments, translated=False
            )
            return [str(dubber.paths.timeline), str(translated), str(source)]

        def save_timeline_ui(data, state, *values):
            try:
                _, dubber, segments, source_language_value = _validated_timeline(data, state, values)
                dubber.save_timeline(segments, source_language_value)
                new_state = {
                    **state,
                    "segments": [item.to_dict() for item in segments],
                    "source_language": source_language_value,
                }
                locked = sum(1 for item in segments if item.translation_locked)
                return (
                    timeline_rows(segments), new_state,
                    f"✅ 时间轴已保存；锁定人工译文 {locked} 行",
                    _export_timeline_files(dubber, segments),
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def export_timeline_ui(data, state, *values):
            try:
                _, dubber, segments, source_language_value = _validated_timeline(data, state, values)
                dubber.save_timeline(segments, source_language_value)
                return _export_timeline_files(dubber, segments), "✅ 已导出 JSON、译文 SRT 和原文 SRT"
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def import_timeline_ui(review_file, data, state, *values):
            try:
                path = _upload_path(review_file)
                if not path:
                    raise ValueError("请先上传人工校对后的时间轴或字幕")
                _, dubber, current, source_language_value = _validated_timeline(data, state, values)
                segments = import_reviewed_timeline(path, current)
                dubber.save_timeline(segments, source_language_value)
                new_state = {
                    **state,
                    "segments": [item.to_dict() for item in segments],
                    "source_language": source_language_value,
                }
                return (
                    timeline_rows(segments), new_state,
                    f"✅ 已导入 {len(segments)} 行人工校对结果并锁定译文",
                    _export_timeline_files(dubber, segments),
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def render_ui(data, state, *values, progress=gr.Progress(track_tqdm=True)):
            try:
                _, dubber, segments, source_language_value = _validated_timeline(data, state, values)
                outputs = dubber.render(
                    segments,
                    progress=lambda value, text: progress(value, desc=text),
                )
                qc = Path(outputs["qc"]).read_text(encoding="utf-8")
                files = [
                    outputs["video"], outputs["audio"], outputs["translated_srt"],
                    outputs["source_srt"], outputs["timeline"], outputs["qc"],
                ]
                new_state = {
                    **state,
                    "segments": [item.to_dict() for item in segments],
                    "source_language": source_language_value,
                }
                return (
                    timeline_rows(segments), new_state, outputs["video"], files,
                    f"```json\n{qc}\n```", "✅ 生成完成",
                    _export_timeline_files(dubber, segments),
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        def one_click_ui(*values, progress=gr.Progress(track_tqdm=True)):
            try:
                config = _make_config(*values)
                dubber = OriginalVoiceDubber(config)
                segments = dubber.analyze(
                    progress=lambda value, text: progress(value * 0.4, desc=text)
                )
                outputs = dubber.render(
                    segments,
                    progress=lambda value, text: progress(0.4 + value * 0.6, desc=text),
                )
                state = {
                    "segments": [item.to_dict() for item in segments],
                    "analysis_key": dubber._analysis_key(),
                    "project_root": str(dubber.paths.root),
                    "source_language": segments[0].source_language or config.source_language,
                }
                qc = Path(outputs["qc"]).read_text(encoding="utf-8")
                files = [
                    outputs["video"], outputs["audio"], outputs["translated_srt"],
                    outputs["source_srt"], outputs["timeline"], outputs["qc"],
                ]
                return (
                    timeline_rows(segments), state, "✅ 一键译制完成", str(dubber.paths.root),
                    outputs["video"], files, f"```json\n{qc}\n```",
                    _export_timeline_files(dubber, segments),
                )
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        refresh_drive.click(
            lambda: gr.update(choices=scan_drive_videos()), outputs=drive_path
        )
        analyze_button.click(
            analyze_ui,
            inputs=common_inputs,
            outputs=[timeline, timeline_state, status, project_path, timeline_export_files],
        )
        render_button.click(
            render_ui,
            inputs=[timeline, timeline_state, *common_inputs],
            outputs=[
                timeline, timeline_state, result_video, result_files, qc_report, status,
                timeline_export_files,
            ],
        )
        save_timeline_button.click(
            save_timeline_ui,
            inputs=[timeline, timeline_state, *common_inputs],
            outputs=[timeline, timeline_state, status, timeline_export_files],
        )
        export_timeline_button.click(
            export_timeline_ui,
            inputs=[timeline, timeline_state, *common_inputs],
            outputs=[timeline_export_files, status],
        )
        import_timeline_button.click(
            import_timeline_ui,
            inputs=[reviewed_timeline_file, timeline, timeline_state, *common_inputs],
            outputs=[timeline, timeline_state, status, timeline_export_files],
        )
        one_click_button.click(
            one_click_ui,
            inputs=common_inputs,
            outputs=[
                timeline, timeline_state, status, project_path, result_video, result_files,
                qc_report, timeline_export_files,
            ],
        )

    return app
