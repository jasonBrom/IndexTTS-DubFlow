from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from shutil import which

import numpy as np
import pytest
import soundfile as sf

import original_dubber.media as media_module
import scripts.translation_worker as translation_worker_module
from original_dubber.asr import (
    apply_hotword_replacements,
    normalize_asr_segments,
    parse_hotword_spec,
    split_long_segments,
)
from original_dubber.indextts_config import (
    inspect_indextts25_config,
    normalize_indextts25_hub_config,
)
from original_dubber.media import (
    build_dialogue_removed_background,
    build_emotion_references,
    build_speaker_references,
    mix_audio,
)
from original_dubber.models import DubConfig, Segment, Word
from original_dubber.pipeline import plan_render_slots
from original_dubber.platforms import runtime_paths, venv_executable
from original_dubber.singing import SingingDetector
from original_dubber.subtitles import import_reviewed_timeline, parse_subtitle_file
from original_dubber.translation import HyMT2Translator, translate_segments
from original_dubber.tts import (
    IndexTTS25Engine,
    _atempo_chain,
    extract_emotion_tag,
    fit_audio_exact,
)
from original_dubber.ui import _make_config
from original_dubber.utils import seconds_to_srt, stable_hash, write_srt


def test_app_bootstrap_does_not_import_gradio_or_torch() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app,sys; print(int('gradio' in sys.modules), int('torch' in sys.modules))",
        ],
        cwd=root,
        check=True,
        encoding="utf-8",
        capture_output=True,
    )
    assert result.stdout.strip() == "0 0"


def test_virtualenv_executable_layouts_are_cross_platform(tmp_path: Path) -> None:
    venv = tmp_path / ".venv"
    assert venv_executable(venv, windows=False) == venv / "bin" / "python"
    assert venv_executable(venv, windows=True) == venv / "Scripts" / "python.exe"


def test_runtime_paths_honor_single_relocatable_root(tmp_path: Path) -> None:
    paths = runtime_paths(tmp_path / "runtime")
    assert paths.indextts == (tmp_path / "runtime" / "index-tts").resolve()
    assert paths.checkpoints == paths.indextts / "checkpoints"
    assert paths.asr == paths.root / "asr-runtimes"
    assert paths.translation == paths.root / "translation-runtimes"


def test_cross_platform_installer_dry_run() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/setup_runtime.py", "--dry-run"],
        cwd=root,
        check=True,
        encoding="utf-8",
        capture_output=True,
    )
    assert "IndexTTS" in result.stdout
    assert ".runtime" in result.stdout


def test_empty_optional_gradio_text_values_are_normalized(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"test-video-placeholder")

    config = _make_config(
        source_mode="输入路径",
        upload=None,
        local_path=str(video),
        drive_path=None,
        project_name=None,
        output_root=None,
        source_language="auto",
        target_language="ZH",
        asr_model="whisper:small",
        input_subtitle=None,
        input_subtitle_mode="不上传字幕，使用 ASR",
        hotword_text=None,
        hotword_file=None,
        translation_backend="HY-MT2-7B 本地（推荐）",
        nllb_model=None,
        hymt2_model=None,
        translation_context_segments=None,
        translation_style=None,
        translation_glossary=None,
        hymt2_quantization=None,
        llm_api_base=None,
        llm_api_key=None,
        llm_model=None,
        separate_background=True,
        protect_singing_vocals=True,
        diarization=False,
        hf_token=None,
        emotion_mode=None,
        emotion_strength=None,
        subtitle_mode=None,
        original_audio_volume=None,
        use_bf16=True,
        use_cuda_kernel=False,
        authorized=True,
    )

    assert config.translation_glossary == ""
    assert config.translation_style == ""
    assert config.asr_hotwords == []
    assert config.llm_api_base == ""
    assert config.hf_token == ""
    assert config.translation_model == "tencent/Hy-MT2-7B"
    assert config.project_name.startswith("dub_input_")
    assert config.translation_context_segments == 3
    assert config.emotion_strength == pytest.approx(0.72)


def test_colab_setup_tracks_official_main_and_includes_qwen_emotion() -> None:
    root = Path(__file__).parents[1]
    script = (root / "scripts" / "colab_setup.sh").read_text(encoding="utf-8")
    assert 'LOG_HEARTBEAT_SECONDS="${LOG_HEARTBEAT_SECONDS:-10}"' in script
    assert 'sleep "${LOG_HEARTBEAT_SECONDS}"' in script
    assert 'PINNED_COMMIT="ccd81054de9859faeb19b773fff0e2e1ae9e959e"' in script
    assert 'qwen0.6bemo4-merge/config.json' in script
    assert 'export INDEXTTS_CONFIG="${INDEXTTS_MODEL_DIR}/config.yaml"' in script
    assert "normalize_indextts25_config.py" in script
    assert "git lfs" not in script
    assert "apt-get update -qq" not in script
    assert "pip install -q" not in script
    assert "pytest>=8" not in script
    assert 'HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"' in script
    assert 'UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"' in script
    assert 'CLEAN_INSTALLER_CACHE="${CLEAN_INSTALLER_CACHE:-1}"' in script
    assert "uv cache clean" in script


def test_hymt_worker_uses_dedicated_cache_and_disables_xet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "hymt-hf"
    monkeypatch.setenv("HYMT_HF_HOME", str(cache))
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    translator = HyMT2Translator(runtime_root=str(tmp_path / "runtime"))

    env = translator._worker_env()

    assert env["HF_HOME"] == str(cache.resolve())
    assert env["HF_HUB_CACHE"] == str((cache / "hub").resolve())
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["PIP_NO_CACHE_DIR"] == "1"


def test_hymt_storage_preflight_removes_only_xet_temp_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hf_home = tmp_path / "huggingface"
    xet_file = hf_home / "xet" / "chunk-cache" / "chunk"
    xet_file.parent.mkdir(parents=True)
    xet_file.write_bytes(b"x" * 4096)
    completed = hf_home / "hub" / "models--unit--tiny" / "blobs" / "finished"
    completed.parent.mkdir(parents=True)
    completed.write_bytes(b"model")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")

    report = translation_worker_module.prepare_model_storage("unit/tiny")

    assert report["removed_xet_gib"] > 0
    assert not (hf_home / "xet").exists()
    assert completed.read_bytes() == b"model"


def _write_v25_hub_config(path: Path) -> None:
    path.write_text(
        """
gpt:
  number_text_tokens: 60509
  condition_type: conformer_perceiver
  condition_module: {input_layer: conv2d2}
  emo_condition_module: {input_layer: conv2d2}
semantic_codec: {codebook_size: 8192}
s2mel:
  length_regulator: {in_channels: 1024}
gpt_checkpoint: /cubefs/internal/gpt.pth
s2mel_checkpoint: /cubefs/internal/s2mel.pth
w2v_stat: wav2vec2bert_stats.pt
emo_matrix: feat2.pt
spk_matrix: feat1.pt
qwen_emo_path: qwen0.6bemo4-merge/
vocoder: {name: bigvgan_generator.pt}
version: 2.0
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_official_v25_hub_config_is_safely_normalized(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_v25_hub_config(config)

    before = inspect_indextts25_config(config)
    report = normalize_indextts25_hub_config(config)
    after = inspect_indextts25_config(config)

    assert before["architecture_is_v25"] is True
    assert before["config_is_v25"] is False
    assert report["changes"]["version"] == {"from": 2.0, "to": 2.5}
    assert after["config_is_v25"] is True
    assert after["has_internal_path"] is False
    assert (tmp_path / "config.hub-original.yaml").is_file()


def test_real_v20_config_is_never_relabelled(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_v25_hub_config(config)
    source = config.read_text(encoding="utf-8").replace(
        "number_text_tokens: 60509", "number_text_tokens: 12000"
    )
    config.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="不具备 IndexTTS 2.5 专属结构"):
        normalize_indextts25_hub_config(config)
    assert "version: 2.0" in config.read_text(encoding="utf-8")


def test_srt_timestamp_rounding() -> None:
    assert seconds_to_srt(0) == "00:00:00,000"
    assert seconds_to_srt(3661.2346) == "01:01:01,235"


def test_segment_duration_and_validation() -> None:
    segment = Segment(id=0, start=1.25, end=3.75, source_text="hello")
    assert segment.duration == 2.5
    segment.validate()
    with pytest.raises(ValueError):
        Segment(id=0, start=3, end=2, source_text="bad").validate()
    with pytest.raises(ValueError):
        Segment(id=0, start=float("nan"), end=2, source_text="bad").validate()


def test_write_srt(tmp_path: Path) -> None:
    segments = [
        Segment(id=0, start=0.1, end=1.2, source_text="hello", target_text="你好"),
        Segment(
            id=1,
            start=2.0,
            end=3.0,
            source_text="off",
            target_text="关闭",
            enabled=False,
        ),
    ]
    output = write_srt(tmp_path / "result.srt", segments)
    text = output.read_text(encoding="utf-8")
    assert "00:00:00,100 --> 00:00:01,200" in text
    assert "你好" in text
    assert "关闭" not in text


@pytest.mark.parametrize(
    ("speed", "minimum", "maximum"),
    [(0.1, 0.5, 2.0), (1.0, 0.5, 2.0), (7.5, 0.5, 2.0)],
)
def test_atempo_chain_is_in_ffmpeg_range(
    speed: float, minimum: float, maximum: float
) -> None:
    chain = _atempo_chain(speed)
    assert all(minimum <= item <= maximum for item in chain)
    product = 1.0
    for item in chain:
        product *= item
    assert product == pytest.approx(speed)


def test_stable_hash() -> None:
    assert stable_hash(["a", 1]) == stable_hash(["a", 1])
    assert stable_hash(["a", 1]) != stable_hash(["a", 2])


def test_patch_contains_absolute_duration_and_sampling_fix() -> None:
    patch = Path(__file__).parents[1] / "patches" / "indextts25_exact_duration.patch"
    text = patch.read_text(encoding="utf-8")
    assert "target_duration_seconds" in text
    assert "do_sample=do_sample" in text
    assert "target_samples" in text
    assert "['ja', 'es', 'ar']" in text


def test_natural_timing_borrows_only_real_following_silence() -> None:
    segments = [
        Segment(id=0, start=0.0, end=1.0, source_text="a"),
        Segment(id=1, start=1.8, end=2.4, source_text="b"),
        Segment(id=2, start=2.35, end=3.0, source_text="overlap"),
    ]
    plan_render_slots(segments, 4.0, guard_seconds=0.05)

    assert segments[0].render_start == 0.0
    assert segments[0].max_render_end == pytest.approx(1.75)
    assert segments[1].max_render_end == pytest.approx(2.4)
    assert segments[2].max_render_end == pytest.approx(3.95)


def test_emotion_tag_is_removed_before_speech() -> None:
    spoken, emotion = extract_emotion_tag("[情感:极度悲伤，声音颤抖] 我没事。")
    assert spoken == "我没事。"
    assert emotion == "极度悲伤，声音颤抖"


def test_short_emotion_reference_uses_same_speaker_not_adjacent_voice(tmp_path: Path) -> None:
    sample_rate = 1000
    audio = np.concatenate(
        [
            np.full(300, 0.10, dtype=np.float32),
            np.full(900, 0.90, dtype=np.float32),
            np.full(1100, 0.20, dtype=np.float32),
        ]
    )
    source = tmp_path / "dialogue.wav"
    sf.write(source, audio, sample_rate)
    segments = [
        Segment(id=0, start=0.0, end=0.3, source_text="a", speaker="A"),
        Segment(id=1, start=0.3, end=1.2, source_text="b", speaker="B"),
        Segment(id=2, start=1.2, end=2.3, source_text="c", speaker="A"),
    ]

    refs = build_emotion_references(
        source, segments, tmp_path / "emotion", sample_rate=sample_rate,
        min_seconds=1.0, max_seconds=1.5,
    )
    rendered, _ = sf.read(refs[0], dtype="float32")

    assert len(rendered) >= sample_rate
    assert float(np.max(np.abs(rendered))) < 0.3


def test_song_vocals_are_only_muted_in_enabled_dialogue_windows(tmp_path: Path) -> None:
    sample_rate = 1000
    instrumental = np.zeros((sample_rate, 2), dtype=np.float32)
    vocals = np.full((sample_rate, 2), 0.25, dtype=np.float32)
    instrumental_path = tmp_path / "instrumental.wav"
    vocals_path = tmp_path / "vocals.wav"
    output_path = tmp_path / "protected_background.wav"
    sf.write(instrumental_path, instrumental, sample_rate)
    sf.write(vocals_path, vocals, sample_rate)
    segments = [
        Segment(id=0, start=0.30, end=0.60, source_text="dialogue", enabled=True),
        Segment(id=1, start=0.70, end=0.90, source_text="song", enabled=False),
    ]

    build_dialogue_removed_background(
        instrumental_path,
        vocals_path,
        segments,
        output_path,
        margin_ms=0,
        fade_ms=0,
    )
    rendered, rendered_rate = sf.read(output_path, dtype="float32", always_2d=True)

    assert rendered_rate == sample_rate
    assert float(np.mean(np.abs(rendered[100:250]))) == pytest.approx(0.25, abs=1e-3)
    assert float(np.max(np.abs(rendered[300:600]))) < 1e-4
    assert float(np.mean(np.abs(rendered[700:900]))) == pytest.approx(0.25, abs=1e-3)


def test_cache_key_tracks_timeline_and_reference_content(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(800, dtype=np.float32), 16_000)
    config = DubConfig(
        source_video=str(tmp_path / "unused.mp4"),
        output_root=str(tmp_path),
        project_name="test",
    )
    engine = IndexTTS25Engine(config)
    segment = Segment(
        id=0,
        start=0.0,
        end=1.0,
        source_text="source",
        target_text="target",
        reference_path=str(reference),
        emotion_reference_path=str(reference),
    )
    initial = engine.segment_cache_key(segment)
    segment.start = 0.1
    shifted = engine.segment_cache_key(segment)
    segment.start = 0.0
    sf.write(reference, np.ones(800, dtype=np.float32) * 0.1, 16_000)
    changed_reference = engine.segment_cache_key(segment)

    assert shifted != initial
    assert changed_reference != initial


def test_singing_label_scoring_is_case_insensitive() -> None:
    labels = {0: "Speech", 1: "Singing", 2: "Music"}
    scores = [0.8, 0.7, 0.9]
    assert SingingDetector._max_score(scores, labels, ("speech",)) == 0.8
    assert SingingDetector._max_score(scores, labels, ("singing", "choir")) == 0.7


@pytest.mark.skipif(which("ffmpeg") is None, reason="ffmpeg is required")
def test_final_limiter_compensates_its_lookahead_latency(tmp_path: Path) -> None:
    sample_rate = 48_000
    impulse = np.zeros(sample_rate, dtype=np.float32)
    impulse[0] = 0.5
    silence = np.zeros(sample_rate, dtype=np.float32)
    background = tmp_path / "background.wav"
    dub = tmp_path / "dub.wav"
    output = tmp_path / "mixed.wav"
    sf.write(background, impulse, sample_rate)
    sf.write(dub, silence, sample_rate)

    mix_audio(background, dub, output)
    rendered, rendered_rate = sf.read(output, dtype="float32")
    nonzero = np.flatnonzero(np.abs(rendered) > 1e-4)

    assert rendered_rate == sample_rate
    assert len(rendered) == sample_rate
    assert len(nonzero) > 0
    assert int(nonzero[0]) <= 1


@pytest.mark.skipif(which("ffmpeg") is None, reason="ffmpeg is required")
def test_legacy_alimiter_fallback_preserves_alignment_and_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_rate = 48_000
    impulse = np.zeros(sample_rate, dtype=np.float32)
    impulse[0] = 0.5
    silence = np.zeros(sample_rate, dtype=np.float32)
    background = tmp_path / "background.wav"
    dub = tmp_path / "dub.wav"
    output = tmp_path / "mixed_legacy.wav"
    sf.write(background, impulse, sample_rate)
    sf.write(dub, silence, sample_rate)
    monkeypatch.setattr(media_module, "_alimiter_supports_latency", lambda: False)

    mix_audio(background, dub, output)
    rendered, rendered_rate = sf.read(output, dtype="float32")
    nonzero = np.flatnonzero(np.abs(rendered) > 1e-4)

    assert rendered_rate == sample_rate
    assert len(rendered) == sample_rate
    assert len(nonzero) > 0
    assert int(nonzero[0]) <= 1


@pytest.mark.skipif(which("ffmpeg") is None, reason="ffmpeg is required")
def test_duration_fallback_finishes_at_the_exact_sample_count(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(22_050, dtype=np.float32), 22_050)

    fit_audio_exact(source, 0.731)
    info = sf.info(source)

    assert info.samplerate == 22_050
    assert info.frames == round(0.731 * 22_050)


def test_long_asr_segments_split_on_word_timestamps() -> None:
    words = [
        Word(start=float(index), end=float(index + 1), text=f" {index}")
        for index in range(20)
    ]
    source = Segment(
        id=0,
        start=0.0,
        end=20.0,
        source_text="long line",
        source_language="en",
        words=words,
    )

    result = split_long_segments([source], max_seconds=6.0)

    assert len(result) >= 3
    assert all(item.duration <= 6.0 for item in result)
    assert [item.id for item in result] == list(range(len(result)))


def test_asr_segmentation_merges_half_sentences_and_prefers_punctuation() -> None:
    fragments = [
        Segment(
            id=0,
            start=0.0,
            end=2.0,
            source_text="这是被切开的",
            words=[Word(0.0, 1.0, "这是"), Word(1.0, 2.0, "被切开的")],
        ),
        Segment(
            id=1,
            start=2.15,
            end=4.0,
            source_text="一句话。下一句很长，仍然继续",
            words=[
                Word(2.15, 3.0, "一句话。"),
                Word(3.0, 3.5, "下一句很长，"),
                Word(3.5, 4.0, "仍然继续"),
            ],
        ),
    ]

    normalized = normalize_asr_segments(fragments, max_seconds=4.5)

    assert normalized[0].source_text == "这是被切开的一句话。"
    assert normalized[0].end == pytest.approx(3.0)
    assert normalized[1].source_text == "下一句很长，仍然继续"


def test_terminal_punctuation_splits_multiple_sentences_without_hard_cut() -> None:
    source = Segment(
        id=0,
        start=0,
        end=8,
        source_text="第一句。第二句。",
        words=[Word(0, 4, "第一句。"), Word(4, 8, "第二句。")],
    )
    result = split_long_segments([source], max_seconds=10)
    assert [item.source_text for item in result] == ["第一句。", "第二句。"]


def test_split_without_word_timestamps_still_respects_limit() -> None:
    source = Segment(
        id=0,
        start=0,
        end=18,
        source_text="这是第一部分，接下来继续说明，最后才结束。",
        source_language="zh",
    )

    result = normalize_asr_segments([source], max_seconds=7)

    assert len(result) >= 3
    assert all(item.duration <= 7.0 + 1e-6 for item in result)
    assert result[0].source_text.endswith("，")


def test_hotword_terms_and_deterministic_replacements() -> None:
    terms, replacements = parse_hotword_spec("IndexTTS\n因得克斯=>IndexTTS,火红")
    assert terms == ["IndexTTS", "火红"]
    assert replacements == {"因得克斯": "IndexTTS"}
    segments = [Segment(id=0, start=0, end=1, source_text="因得克斯 二点五")]
    apply_hotword_replacements(segments, replacements)
    assert segments[0].source_text == "IndexTTS 二点五"


def test_translation_only_fills_unlocked_empty_rows() -> None:
    locked = Segment(
        id=0,
        start=0,
        end=1,
        source_text="source",
        target_text="人工译文",
        translation_locked=True,
    )
    pending = Segment(id=1, start=1, end=2, source_text="second")
    translate_segments(
        [locked, pending], backend="none", source_language="en", target_language="ZH"
    )
    assert locked.target_text == "人工译文"
    assert pending.target_text == "second"


def test_uploaded_translated_srt_is_locked_and_preserves_timing(tmp_path: Path) -> None:
    subtitle = tmp_path / "manual.srt"
    subtitle.write_text(
        "1\n00:00:01,250 --> 00:00:03,500\n手工字幕\n",
        encoding="utf-8",
    )
    segments = parse_subtitle_file(
        subtitle, mode="translated", source_language="auto", target_language="ZH"
    )
    assert len(segments) == 1
    assert segments[0].start == pytest.approx(1.25)
    assert segments[0].end == pytest.approx(3.5)
    assert segments[0].target_text == "手工字幕"
    assert segments[0].translation_locked is True


def test_uploaded_ass_is_parsed_without_style_markup(tmp_path: Path) -> None:
    subtitle = tmp_path / "manual.ass"
    subtitle.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,{\\i1}Hello\\Nworld\n",
        encoding="utf-8",
    )
    segments = parse_subtitle_file(
        subtitle, mode="source", source_language="en", target_language="ZH"
    )
    assert segments[0].source_text == "Hello world"
    assert segments[0].target_text == ""
    assert segments[0].translation_locked is False


def test_uploaded_vtt_accepts_timestamps_without_hour(tmp_path: Path) -> None:
    subtitle = tmp_path / "manual.vtt"
    subtitle.write_text("WEBVTT\n\n00:01.200 --> 00:02.800\nHello\n", encoding="utf-8")
    segments = parse_subtitle_file(
        subtitle, mode="source", source_language="en", target_language="ZH"
    )
    assert segments[0].start == pytest.approx(1.2)
    assert segments[0].end == pytest.approx(2.8)


def test_reviewed_srt_reimports_and_locks_manual_translation(tmp_path: Path) -> None:
    current = [
        Segment(id=0, start=1.2, end=2.8, source_text="Hello", target_text="旧译文")
    ]
    reviewed = tmp_path / "reviewed.srt"
    reviewed.write_text(
        "1\n00:00:01,200 --> 00:00:02,800\n人工校对译文\n", encoding="utf-8"
    )

    result = import_reviewed_timeline(reviewed, current)

    assert result[0].target_text == "人工校对译文"
    assert result[0].translation_locked is True
    assert result[0].generated_path == ""


def test_speaker_reference_names_do_not_collide_after_sanitizing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dialogue.wav"
    sf.write(source, np.ones(32_000, dtype=np.float32) * 0.05, 16_000)
    segments = [
        Segment(id=0, start=0.0, end=1.0, source_text="a", speaker="A/B"),
        Segment(id=1, start=1.0, end=2.0, source_text="b", speaker="A?B"),
    ]

    references = build_speaker_references(source, segments, tmp_path / "refs")

    assert len({str(path) for path in references.values()}) == 2
    assert all(path.is_file() for path in references.values())
