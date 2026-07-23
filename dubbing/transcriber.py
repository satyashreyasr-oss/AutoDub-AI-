from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from .utils import log


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    language: str
    segments: list[Segment]


def run_whisper(audio_path: Path, model_size: str, device: str, task: str) -> TranscriptionResult:
    resolved_device = "cuda" if device in ("auto", "cuda") else "cpu"
    compute_type = "float16" if resolved_device == "cuda" else "int8"

    log("whisper", f"Loading model '{model_size}' on {resolved_device} ({compute_type})")
    try:
        model = WhisperModel(model_size, device=resolved_device, compute_type=compute_type)
    except Exception as exc:
        if resolved_device == "cuda":
            log("whisper", f"CUDA unavailable ({exc}); falling back to CPU")
            resolved_device = "cpu"
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
            raise

    log("whisper", f"Running task='{task}' on {audio_path.name}")
    segments_iter, info = model.transcribe(str(audio_path), task=task, vad_filter=True)

    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        segments.append(Segment(start=seg.start, end=seg.end, text=text))
        log("whisper", f"[{seg.start:7.1f}s - {seg.end:7.1f}s] {text}")

    log(
        "whisper",
        f"Detected language: {info.language} (p={info.language_probability:.2f}), {len(segments)} segments",
    )
    return TranscriptionResult(language=info.language, segments=segments)
