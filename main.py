import argparse
import sys
from pathlib import Path

from dubbing.pipeline import DubbingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dub a YouTube video into English.")
    parser.add_argument("url", nargs="?", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output-dir", default="output", help="Directory for downloaded and dubbed files"
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="faster-whisper model size: tiny, base, small, medium, large-v3",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for Whisper inference",
    )
    parser.add_argument(    
        "--voice", default=None, help="Force a specific edge-tts voice (e.g. en-US-GuyNeural)"
    )
    parser.add_argument(
        "--translator",
        default="whisper",
        choices=["whisper", "indictrans2"],
        help="Translation backend (indictrans2 only applies to Indian-language sources)",
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep intermediate files instead of cleaning up"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = args.url or input("YouTube URL: ").strip()
    if not url:
        print("No URL provided.")
        sys.exit(1)

    pipeline = DubbingPipeline(
        output_dir=Path(args.output_dir),
        whisper_model=args.whisper_model,
        device=args.device,
        voice=args.voice,
        translator_backend=args.translator,
        keep_temp=args.keep_temp,
    )
    pipeline.run(url)


if __name__ == "__main__":
    main()
