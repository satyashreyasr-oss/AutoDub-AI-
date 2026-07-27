from pathlib import Path

from .transcriber import Segment
from .utils import log


def format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments: list[Segment], path: Path) -> Path:
    blocks = []
    index = 1
    for seg in segments:
        if not seg.text:
            continue
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}\n"
            f"{seg.text}\n"
        )
        index += 1

    path.write_text("\n".join(blocks), encoding="utf-8")
    log("subtitles", f"Wrote {index - 1} subtitle cues -> {path}")
    return path
