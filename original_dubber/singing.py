from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .models import Segment
from .utils import release_cuda

SINGING_LABELS = (
    "singing",
    "choir",
    "vocal music",
    "rapping",
    "humming",
    "yodeling",
    "chant",
    "mantra",
)
SPEECH_LABELS = (
    "speech",
    "conversation",
    "narration",
    "monologue",
    "child speech",
)


class SingingDetector:
    """AudioSet AST classifier used to keep song vocals out of the dubbing timeline."""

    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path
        self.extractor = None
        self.model = None
        self.device = "cpu"

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import ASTForAudioClassification, AutoFeatureExtractor
        except ImportError as exc:
            raise RuntimeError("当前 IndexTTS 环境缺少 AST 音频分类支持。") from exc
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        local_only = Path(self.model_name_or_path).is_dir()
        self.extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name_or_path, local_files_only=local_only
        )
        self.model = ASTForAudioClassification.from_pretrained(
            self.model_name_or_path, local_files_only=local_only
        ).to(self.device).eval()

    @staticmethod
    def _max_score(scores: list[float], labels: dict[int, str], needles: tuple[str, ...]) -> float:
        values = [
            score
            for index, score in enumerate(scores)
            if any(needle in labels.get(index, "").lower() for needle in needles)
        ]
        return max(values, default=0.0)

    def classify(self, audio, sample_rate: int = 16_000) -> tuple[float, float]:
        self.load()
        import torch

        inputs = self.extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            scores = torch.sigmoid(self.model(**inputs).logits)[0].float().cpu().tolist()
        labels = {int(key): str(value) for key, value in self.model.config.id2label.items()}
        speech = self._max_score(scores, labels, SPEECH_LABELS)
        singing = self._max_score(scores, labels, SINGING_LABELS)
        return speech, singing

    def unload(self) -> None:
        self.extractor = None
        self.model = None
        release_cuda()


def mark_singing_segments(
    audio_path: str | Path,
    segments: list[Segment],
    *,
    model_name_or_path: str,
    progress: Callable[[float, str], None] | None = None,
    threshold: float = 0.16,
    singing_over_speech: float = 1.08,
) -> list[Segment]:
    import librosa

    if progress:
        progress(0.0, "读取 vocal stem 并下载/加载 AST 唱歌检测器")
    audio, sample_rate = librosa.load(str(audio_path), sr=16_000, mono=True)
    detector = SingingDetector(model_name_or_path)
    try:
        detector.load()
        if progress:
            progress(0.02, "AST 唱歌检测器已加载")
        for index, segment in enumerate(segments):
            padding = 0.25
            start = max(0, int((segment.start - padding) * sample_rate))
            end = min(len(audio), int((segment.end + padding) * sample_rate))
            clip = audio[start:end]
            if len(clip) < int(0.45 * sample_rate):
                continue
            speech_score, singing_score = detector.classify(clip, sample_rate)
            segment.speech_score = round(speech_score, 4)
            segment.singing_score = round(singing_score, 4)
            if singing_score >= threshold and singing_score > speech_score * singing_over_speech:
                segment.content_type = "singing"
                segment.enabled = False
                segment.warning = (
                    f"检测为歌曲人声，默认保留原声、不翻译（唱歌 {singing_score:.2f} / 说话 {speech_score:.2f}）"
                )
            if progress:
                progress((index + 1) / max(1, len(segments)), f"区分对白/唱歌 {index + 1}/{len(segments)}")
    finally:
        detector.unload()
    return segments
