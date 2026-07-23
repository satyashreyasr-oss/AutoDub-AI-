import asyncio
import time
from pathlib import Path

import edge_tts
from pydub import AudioSegment

from .transcriber import TranscriptionResult
from .utils import log, run

MIN_TEMPO = 0.6
MAX_TEMPO = 1.7
TEMPO_TOLERANCE = 0.03
MAX_TTS_ATTEMPTS = 4
TTS_RETRY_BASE_DELAY = 1.5


async def _synthesize_segment(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def _synthesize_segment_with_retry(text: str, voice: str, out_path: Path) -> bool:
    for attempt in range(1, MAX_TTS_ATTEMPTS + 1):
        try:
            asyncio.run(_synthesize_segment(text, voice, out_path))
            return True
        except Exception as exc:
            if attempt == MAX_TTS_ATTEMPTS:
                log("synthesize", f"Skipping segment after {attempt} failed attempts: {exc}")
                return False
            delay = TTS_RETRY_BASE_DELAY * attempt
            log("synthesize", f"TTS attempt {attempt} failed ({exc}); retrying in {delay:.1f}s")
            time.sleep(delay)
    return False


def _apply_tempo(src: Path, dst: Path, tempo: float) -> None:
    run(["ffmpeg", "-y", "-i", str(src), "-filter:a", f"atempo={tempo:.3f}", str(dst)])


def synthesize_dubbed_audio(
    transcription: TranscriptionResult, voice: str, total_duration: float, work_dir: Path
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    track_length_ms = int(total_duration * 1000) + 1000
    track = AudioSegment.silent(duration=track_length_ms)

    for i, seg in enumerate(transcription.segments):
        if not seg.text:
            continue

        raw_path = work_dir / f"seg_{i:04d}_raw.mp3"
        if not _synthesize_segment_with_retry(seg.text, voice, raw_path):
            continue

        raw_audio = AudioSegment.from_file(raw_path)
        raw_duration = len(raw_audio) / 1000.0
        target_duration = max(seg.end - seg.start, 0.1)
        tempo = max(MIN_TEMPO, min(MAX_TEMPO, raw_duration / target_duration))

        if abs(tempo - 1.0) > TEMPO_TOLERANCE:
            tuned_path = work_dir / f"seg_{i:04d}_tuned.mp3"
            _apply_tempo(raw_path, tuned_path, tempo)
            seg_audio = AudioSegment.from_file(tuned_path)
        else:
            seg_audio = raw_audio

        position_ms = int(seg.start * 1000)
        track = track.overlay(seg_audio, position=position_ms)
        log("synthesize", f"[{seg.start:7.1f}s] tempo={tempo:.2f} '{seg.text[:60]}'")

    output_path = work_dir / "dubbed_audio.wav"
    track.export(output_path, format="wav")
    log("synthesize", f"Assembled dubbed audio track -> {output_path}")
    return output_path
