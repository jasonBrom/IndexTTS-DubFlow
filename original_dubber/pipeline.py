from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .asr import assign_speakers_pyannote, create_asr, normalize_asr_segments
from .media import (
    assemble_vocal_track,
    build_dialogue_removed_background,
    build_emotion_references,
    build_speaker_references,
    extract_audio,
    mix_audio,
    mux_video,
    probe_media,
    separate_vocals,
)
from .models import DubConfig, Segment
from .singing import mark_singing_segments
from .subtitles import parse_subtitle_file
from .translation import translate_segments
from .tts import IndexTTS25Engine
from .utils import (
    atomic_write_json,
    file_fingerprint,
    read_json,
    safe_name,
    stable_hash,
    write_srt,
)

Progress = Callable[[float, str], None]


def _noop_progress(_: float, __: str) -> None:
    return None


def plan_render_slots(
    segments: list[Segment],
    total_duration: float,
    *,
    guard_seconds: float = 0.04,
) -> list[Segment]:
    """Keep each source start and lend only genuine following silence to it."""
    ordered = sorted(segments, key=lambda item: (item.start, item.end, item.id))
    for index, segment in enumerate(ordered):
        segment.render_start = segment.start
        segment.render_end = None
        next_start = ordered[index + 1].start if index + 1 < len(ordered) else total_duration
        if next_start >= segment.end:
            maximum_end = max(segment.end, next_start - guard_seconds)
        else:
            maximum_end = segment.end
        segment.max_render_end = min(total_duration, max(segment.end, maximum_end))
    return ordered


class ProjectPaths:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.input = self.root / "input"
        self.audio = self.root / "audio"
        self.stems = self.root / "stems"
        self.refs = self.root / "speakers"
        self.emo_refs = self.root / "emotion_refs"
        self.tts = self.root / "tts"
        self.subtitles = self.root / "subtitles"
        self.mix = self.root / "mix"
        self.output = self.root / "output"
        self.manifest = self.root / "manifest.json"
        self.timeline = self.root / "timeline.json"
        self.qc = self.root / "qc_report.json"
        for path in (
            self.input,
            self.audio,
            self.stems,
            self.refs,
            self.emo_refs,
            self.tts,
            self.subtitles,
            self.mix,
            self.output,
        ):
            path.mkdir(parents=True, exist_ok=True)


class OriginalVoiceDubber:
    def __init__(self, config: DubConfig) -> None:
        config.validate()
        self.config = config
        project_root = Path(config.output_root).expanduser() / safe_name(config.project_name)
        self.paths = ProjectPaths(project_root)
        source = Path(config.source_video).expanduser().resolve()
        if any(part.lower().startswith("gradio") for part in source.parts) or str(source).startswith("/tmp/"):
            persisted = self.paths.input / ("source" + source.suffix.lower())
            if not persisted.is_file() or file_fingerprint(persisted) != file_fingerprint(source):
                shutil.copy2(source, persisted)
            self.config.source_video = str(persisted)
        if config.input_subtitle_path:
            subtitle = Path(config.input_subtitle_path).expanduser().resolve()
            if any(part.lower().startswith("gradio") for part in subtitle.parts) or str(subtitle).startswith("/tmp/"):
                persisted_subtitle = self.paths.input / ("subtitles" + subtitle.suffix.lower())
                if (
                    not persisted_subtitle.is_file()
                    or file_fingerprint(persisted_subtitle) != file_fingerprint(subtitle)
                ):
                    shutil.copy2(subtitle, persisted_subtitle)
                self.config.input_subtitle_path = str(persisted_subtitle)
        self.video_info: dict = {}
        self.dialogue_audio: Path | None = None
        self.background_audio: Path | None = None
        self.original_audio: Path | None = None
        self.vocal_stem: Path | None = None

    def _analysis_key(self) -> str:
        return stable_hash(
            [
                file_fingerprint(self.config.source_video),
                self.config.source_language,
                self.config.target_language,
                self.config.asr_model,
                self.config.asr_hotwords,
                self.config.asr_replacements,
                self.config.input_subtitle_mode,
                (
                    file_fingerprint(self.config.input_subtitle_path)
                    if self.config.input_subtitle_path
                    else ""
                ),
                self.config.min_segment_seconds,
                self.config.max_segment_seconds,
                self.config.translation_backend,
                self.config.translation_model,
                self.config.translation_context_segments,
                self.config.translation_style,
                self.config.translation_glossary,
                self.config.hymt2_quantization,
                self.config.llm_api_base,
                self.config.llm_model,
                self.config.index_api_base,
                self.config.index_model,
                self.config.index_syllables_per_second,
                self.config.index_max_tokens,
                self.config.separate_background,
                self.config.protect_singing_vocals,
                self.config.singing_model,
                self.config.diarization,
            ]
        )

    def _media_key(self) -> str:
        return stable_hash(
            [
                file_fingerprint(self.config.source_video),
                self.config.separate_background,
                "demucs-htdemucs-two-stems-vocals-v1",
            ]
        )

    def _load_manifest_media(self) -> bool:
        manifest = read_json(self.paths.manifest, {})
        if not manifest:
            return False
        if manifest.get("media_key") != self._media_key():
            return False
        self.video_info = manifest.get("video_info", {})
        for attribute, key in (
            ("dialogue_audio", "dialogue_audio"),
            ("background_audio", "background_audio"),
            ("original_audio", "original_audio"),
        ):
            value = manifest.get(key)
            if not value or not Path(value).is_file():
                return False
            setattr(self, attribute, Path(value))
        vocal_stem = manifest.get("vocal_stem")
        if self.config.separate_background:
            if not vocal_stem or not Path(vocal_stem).is_file():
                return False
            self.vocal_stem = Path(vocal_stem)
        else:
            self.vocal_stem = self.original_audio
        return True

    def prepare_media(self, progress: Progress = _noop_progress) -> None:
        if self.config.resume and self._load_manifest_media():
            progress(1.0, "复用已提取的媒体与分轨")
            return
        progress(0.05, "读取视频信息")
        input_fingerprint = file_fingerprint(self.config.source_video)
        self.video_info = probe_media(self.config.source_video)
        self.original_audio = extract_audio(
            self.config.source_video,
            self.paths.audio / "original.wav",
            sample_rate=44_100,
            channels=2,
        )
        if self.config.separate_background:
            progress(0.15, "Demucs 分离对白与背景（首次会下载模型）")
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            vocals, background = separate_vocals(
                self.original_audio, self.paths.stems / input_fingerprint, device=device
            )
            self.vocal_stem = vocals
            self.dialogue_audio = self.vocal_stem
            self.background_audio = background
        else:
            self.dialogue_audio = self.original_audio
            self.background_audio = self.original_audio
            self.vocal_stem = self.original_audio
        dialogue_16k = extract_audio(
            self.dialogue_audio,
            self.paths.audio / "dialogue_16k.wav",
            sample_rate=16_000,
            channels=1,
        )
        self.dialogue_audio = dialogue_16k
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "input_path": str(Path(self.config.source_video).resolve()),
            "input_fingerprint": input_fingerprint,
            "media_key": self._media_key(),
            "video_info": self.video_info,
            "dialogue_audio": str(self.dialogue_audio),
            "background_audio": str(self.background_audio),
            "original_audio": str(self.original_audio),
            "vocal_stem": str(self.vocal_stem),
            "config": self.config.public_dict(),
            "indextts_commit": "ccd81054de9859faeb19b773fff0e2e1ae9e959e",
        }
        atomic_write_json(self.paths.manifest, manifest)
        progress(1.0, "媒体准备完成")

    def _load_timeline(self, expected_analysis_key: str | None = None) -> list[Segment]:
        value = read_json(self.paths.timeline, {})
        if expected_analysis_key is not None and value.get("analysis_key") != expected_analysis_key:
            return []
        return [Segment.from_dict(item) for item in value.get("segments", [])]

    def save_timeline(self, segments: list[Segment], source_language: str) -> None:
        atomic_write_json(
            self.paths.timeline,
            {
                "source_language": source_language,
                "target_language": self.config.target_language,
                "analysis_key": self._analysis_key(),
                "segments": [item.to_dict() for item in segments],
            },
        )

    @staticmethod
    def _restore_locked_translations(
        segments: list[Segment], previous: list[Segment]
    ) -> list[Segment]:
        locked = [item for item in previous if item.translation_locked and item.target_text.strip()]
        used: set[int] = set()
        for segment in segments:
            best: tuple[float, int, Segment] | None = None
            for index, old in enumerate(locked):
                if index in used:
                    continue
                overlap = max(0.0, min(segment.end, old.end) - max(segment.start, old.start))
                ratio = overlap / max(0.001, min(segment.duration, old.duration))
                same_text = "".join(segment.source_text.split()) == "".join(old.source_text.split())
                if ratio < 0.85 or (not same_text and abs(segment.start - old.start) > 0.12):
                    continue
                score = ratio + (1.0 if same_text else 0.0)
                if best is None or score > best[0]:
                    best = (score, index, old)
            if best is not None:
                _, index, old = best
                used.add(index)
                segment.target_text = old.target_text
                segment.translation_locked = True
        return segments

    def analyze(self, progress: Progress = _noop_progress, force: bool = False) -> list[Segment]:
        self.prepare_media(lambda value, text: progress(value * 0.25, text))
        previous = self._load_timeline()
        existing = self._load_timeline(self._analysis_key())
        if existing and self.config.resume and not force:
            progress(1.0, "已载入可继续编辑的时间轴")
            return existing
        if self.config.input_subtitle_mode != "none":
            progress(0.28, "读取上传字幕；跳过 ASR")
            segments = parse_subtitle_file(
                self.config.input_subtitle_path,
                mode=self.config.input_subtitle_mode,
                source_language=self.config.source_language,
                target_language=self.config.target_language,
            )
            source_language = (
                self.config.target_language.lower()
                if self.config.input_subtitle_mode == "translated"
                else self.config.source_language
            )
        else:
            progress(0.28, "开始语音识别")
            asr = create_asr(
                self.config.asr_model,
                runtime_root=self.config.asr_runtime_root,
                model_source=self.config.model_source,
            )
            result = asr.transcribe(
                self.dialogue_audio,
                source_language=self.config.source_language,
                min_segment_seconds=self.config.min_segment_seconds,
                hotwords=self.config.asr_hotwords,
                replacements=self.config.asr_replacements,
                progress=lambda value, text: progress(0.28 + value * 0.25, text),
            )
            segments = normalize_asr_segments(result.segments, self.config.max_segment_seconds)
            source_language = result.language
            asr.unload()
        if not segments:
            raise RuntimeError("没有识别到有效对白。请检查视频音轨或关闭人声分离后重试。")
        segments = self._restore_locked_translations(segments, previous)
        if self.config.separate_background and self.config.protect_singing_vocals:
            progress(0.51, "检测歌曲演唱，避免把背景歌声当对白翻译")
            local_ast = Path(self.config.model_dir) / "hf_cache" / "ast-audioset"
            singing_model = str(local_ast) if local_ast.is_dir() else self.config.singing_model
            segments = mark_singing_segments(
                self.dialogue_audio,
                segments,
                model_name_or_path=singing_model,
                progress=lambda value, text: progress(0.51 + value * 0.08, text),
            )
        if self.config.diarization:
            progress(0.60, "说话人分离")
            segments = assign_speakers_pyannote(
                self.dialogue_audio, segments, hf_token=self.config.hf_token
            )
        progress(0.66, "翻译并按原片时长约束措辞")
        active_segments = [segment for segment in segments if segment.enabled]
        if active_segments:
            translate_segments(
                active_segments,
                backend=self.config.translation_backend,
                source_language=source_language,
                target_language=self.config.target_language,
                model_name=self.config.translation_model,
                api_base=self.config.llm_api_base,
                api_key=self.config.llm_api_key,
                api_model=self.config.llm_model,
                runtime_root=self.config.translation_runtime_root,
                context_size=self.config.translation_context_segments,
                style=self.config.translation_style,
                glossary=self.config.translation_glossary,
                hymt2_quantization=self.config.hymt2_quantization,
                index_api_base=self.config.index_api_base,
                index_api_key=self.config.index_api_key,
                index_model=self.config.index_model,
                index_syllables_per_second=self.config.index_syllables_per_second,
                index_max_tokens=self.config.index_max_tokens,
                progress=lambda value, text: progress(0.66 + value * 0.18, text),
            )
        progress(0.86, "建立说话人音色与逐句情感参考")
        references = build_speaker_references(
            self.dialogue_audio,
            segments,
            self.paths.refs,
            max_seconds=self.config.max_reference_seconds,
        )
        emotions = build_emotion_references(
            self.dialogue_audio,
            segments,
            self.paths.emo_refs,
            max_seconds=self.config.max_emotion_reference_seconds,
        )
        for segment in segments:
            if segment.enabled:
                segment.reference_path = str(references[segment.speaker])
                if self.config.emotion_mode == "segment":
                    segment.emotion_reference_path = str(emotions[segment.id])
                else:
                    segment.emotion_reference_path = segment.reference_path
            segment.target_language = self.config.target_language
            segment.validate()
        self.save_timeline(segments, source_language)
        write_srt(self.paths.subtitles / "source.srt", segments, translated=False)
        write_srt(self.paths.subtitles / "translated.srt", segments, translated=True)
        progress(1.0, f"分析完成，共 {len(segments)} 段")
        return segments

    def _refresh_references(self, segments: list[Segment]) -> None:
        references = build_speaker_references(
            self.dialogue_audio,
            segments,
            self.paths.refs,
            max_seconds=self.config.max_reference_seconds,
        )
        emotions = build_emotion_references(
            self.dialogue_audio,
            segments,
            self.paths.emo_refs,
            max_seconds=self.config.max_emotion_reference_seconds,
        )
        for segment in segments:
            if segment.enabled:
                segment.reference_path = str(references[segment.speaker])
                segment.emotion_reference_path = (
                    str(emotions[segment.id])
                    if self.config.emotion_mode == "segment"
                    else segment.reference_path
                )

    def render(
        self,
        segments: list[Segment] | None = None,
        progress: Progress = _noop_progress,
    ) -> dict[str, str]:
        self.prepare_media(lambda value, text: progress(value * 0.05, text))
        if segments is None:
            segments = self._load_timeline(self._analysis_key())
        if not segments:
            raise RuntimeError("时间轴为空，请先执行“分析与翻译”。")
        total_duration = float(
            self.video_info.get("duration") or probe_media(self.config.source_video)["duration"]
        )
        for index, segment in enumerate(segments):
            segment.target_language = self.config.target_language
            segment.validate()
            if segment.end > total_duration + 0.05:
                raise ValueError(
                    f"第 {index + 1} 行结束时间 {segment.end:.3f}s 超出视频时长 {total_duration:.3f}s"
                )
            if segment.end > total_duration:
                segment.end = total_duration
                segment.validate()
            if segment.duration > self.config.max_segment_seconds + 1e-6:
                raise ValueError(
                    f"第 {index + 1} 行时长 {segment.duration:.3f}s 超过安全上限 "
                    f"{self.config.max_segment_seconds:.1f}s，请拆成更短片段"
                )
            if segment.enabled and not segment.target_text.strip():
                raise ValueError(f"第 {index + 1} 行译文为空")
        segments.sort(key=lambda item: (item.start, item.end, item.id))
        for index, segment in enumerate(segments):
            segment.id = index
        plan_render_slots(segments, total_duration)
        self._refresh_references(segments)
        timeline_meta = read_json(self.paths.timeline, {})
        source_language = timeline_meta.get("source_language", self.config.source_language)
        self.save_timeline(segments, source_language)

        engine = IndexTTS25Engine(self.config)
        enabled = [item for item in segments if item.enabled]
        try:
            for offset, segment in enumerate(enabled):
                progress(0.08 + 0.72 * offset / max(1, len(enabled)), f"合成 {offset + 1}/{len(enabled)}")
                output = self.paths.tts / f"segment_{segment.id:05d}.wav"
                cache = self.paths.tts / f"segment_{segment.id:05d}.json"
                cache_value = read_json(cache, {})
                cache_key = engine.segment_cache_key(segment)
                if self.config.resume and output.is_file() and cache_value.get("key") == cache_key:
                    restored = cache_value.get("segment")
                    if restored:
                        cached_segment = Segment.from_dict(restored)
                        segment.generated_path = str(output)
                        segment.natural_duration = cached_segment.natural_duration
                        segment.duration_factor = cached_segment.duration_factor
                        segment.render_start = cached_segment.render_start
                        segment.render_end = cached_segment.render_end
                        if cached_segment.warning:
                            segment.warning = cached_segment.warning
                    continue
                engine.load(
                    progress=lambda text, current_offset=offset: progress(
                        0.08 + 0.72 * current_offset / max(1, len(enabled)), text
                    )
                )
                segment = engine.synthesize_segment(segment, output)
                segments[segment.id] = segment
                atomic_write_json(cache, {"key": cache_key, "segment": segment.to_dict()})
                self.save_timeline(segments, source_language)
        finally:
            engine.unload()

        progress(0.82, "按绝对时间轴混合译制对白")
        dubbed_track = assemble_vocal_track(
            segments,
            self.paths.mix / "dubbed_vocals.wav",
            total_duration=total_duration,
            gain=self.config.dub_volume,
        )
        render_background = self.background_audio
        if self.config.separate_background:
            progress(0.86, "只移除对白，回填歌曲演唱与其他非对白人声")
            render_background = build_dialogue_removed_background(
                self.background_audio,
                self.vocal_stem,
                segments,
                self.paths.mix / "background_with_song_vocals.wav",
                margin_ms=self.config.dialogue_mask_margin_ms,
            )
        background_volume = (
            self.config.background_volume
            if self.config.separate_background
            else self.config.original_audio_volume
        )
        mixed = mix_audio(
            render_background,
            dubbed_track,
            self.paths.mix / "mixed_audio.wav",
            background_volume=background_volume,
            dub_volume=1.0,
        )
        translated_srt = write_srt(
            self.paths.subtitles / "translated.srt", segments, translated=True
        )
        source_srt = write_srt(self.paths.subtitles / "source.srt", segments, translated=False)
        progress(0.92, "封装视频、音轨与字幕")
        output_video = mux_video(
            self.config.source_video,
            mixed,
            self.paths.output / f"{safe_name(self.config.project_name)}_{self.config.target_language}.mp4",
            subtitle_path=translated_srt if self.config.subtitle_mode != "none" else None,
            subtitle_language=self.config.target_language.lower(),
            subtitle_mode=self.config.subtitle_mode,
        )
        warnings = [item.warning for item in segments if item.warning]
        qc = {
            "segments": len(segments),
            "enabled_segments": len(enabled),
            "singing_segments_preserved": sum(
                1 for item in segments if item.content_type == "singing" and not item.enabled
            ),
            "warning_count": len(warnings),
            "warnings": warnings,
            "video_duration": total_duration,
            "target_language": self.config.target_language,
            "indextts_commit": "ccd81054de9859faeb19b773fff0e2e1ae9e959e",
            "natural_speed_segments": sum(
                1 for item in enabled if item.duration_factor is not None and abs(item.duration_factor - 1.0) < 1e-6
            ),
            "compressed_segments": sum(
                1 for item in enabled if item.duration_factor is not None and item.duration_factor < 0.999
            ),
            "note": (
                "每句保持原起点，自然语速优先；只借用下一句前的空白。放不下时才使用 "
                "IndexTTS 2.5 S2M 目标帧控制，最终仅做样本级收尾。"
            ),
        }
        atomic_write_json(self.paths.qc, qc)
        self.save_timeline(segments, source_language)
        progress(1.0, "译制视频完成")
        return {
            "video": str(output_video),
            "audio": str(mixed),
            "translated_srt": str(translated_srt),
            "source_srt": str(source_srt),
            "timeline": str(self.paths.timeline),
            "qc": str(self.paths.qc),
            "project": str(self.paths.root),
        }
