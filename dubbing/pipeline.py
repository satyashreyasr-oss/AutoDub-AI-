import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .downloader import download_video
from .muxer import remux_audio
from .synthesizer import synthesize_dubbed_audio
from .translator import get_english_segments
from .utils import format_duration, log, run
from .voice_picker import pick_voice


@dataclass
class DubbingPipeline:
    output_dir: Path
    whisper_model: str = "small"
    device: str = "auto"
    voice: str | None = None
    translator_backend: str = "whisper"
    keep_temp: bool = False

    def run(self, url: str) -> Path:
        start_time = time.monotonic()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        download = download_video(url, self.output_dir)
        work_dir = self.output_dir / f"{download.video_path.stem}_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        audio_path = work_dir / "source_audio.wav"
        log("extract", "Extracting audio track for analysis")
        run(
            [
                "ffmpeg", "-y",
                "-i", str(download.video_path),
                "-vn", "-ac", "1", "-ar", "16000",
                str(audio_path),
            ]
        )

        transcription = get_english_segments(
            audio_path, self.whisper_model, self.device, self.translator_backend
        )
        voice = pick_voice(audio_path, self.voice)
        dubbed_audio = synthesize_dubbed_audio(
            transcription, voice, download.duration, work_dir
        )

        output_path = self.output_dir / f"{download.video_path.stem}_dubbed_en.mp4"
        remux_audio(download.video_path, dubbed_audio, output_path)

        if not self.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)

        elapsed = time.monotonic() - start_time
        log("done", f"Finished in {format_duration(elapsed)} -> {output_path}")
        return output_path
