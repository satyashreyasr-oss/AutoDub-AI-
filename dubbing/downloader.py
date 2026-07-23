from dataclasses import dataclass
from pathlib import Path

import yt_dlp

from .utils import log


@dataclass
class DownloadResult:
    video_path: Path
    title: str
    duration: float


def download_video(url: str, output_dir: Path) -> DownloadResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    log("download", f"Fetching video info for {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info["id"]
    video_path = output_dir / f"{video_id}.mp4"
    if not video_path.exists():
        candidates = sorted(output_dir.glob(f"{video_id}.*"))
        if not candidates:
            raise FileNotFoundError(f"Could not locate downloaded file for video id {video_id}")
        video_path = candidates[0]

    duration = float(info.get("duration") or 0.0)
    title = info.get("title", video_id)
    log("download", f"Downloaded '{title}' ({duration:.0f}s) -> {video_path}")
    return DownloadResult(video_path=video_path, title=title, duration=duration)
