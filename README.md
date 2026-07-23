# Automated Video Dubbing System

Turns a YouTube video in any language into an English-dubbed version: same video,
same speaker energy, new English audio track.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requires `ffmpeg` on PATH (already available on this machine). A CUDA GPU is
used automatically if present; otherwise everything falls back to CPU.

## Usage

```
python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `-o, --output-dir` | `output` | Where downloads and final videos are written |
| `--whisper-model` | `small` | faster-whisper size: tiny/base/small/medium/large-v3 |
| `--device` | `auto` | `cpu` / `cuda` / `auto` |
| `--voice` | auto-detected | force an edge-tts voice, e.g. `en-US-GuyNeural` |
| `--translator` | `whisper` | `whisper` (default, single-pass) or `indictrans2` (Indian languages only) |
| `--keep-temp` | off | keep intermediate segment/audio files for debugging |

## Architecture

```
YouTube URL
  -> downloader.py      yt-dlp: fetch best video+audio, merge to mp4
  -> pipeline.py         ffmpeg: extract mono 16kHz wav for analysis
  -> translator.py       faster-whisper (task=translate): source speech -> English
                          text + segment timestamps, in one pass
                          [optional: task=transcribe + IndicTrans2 for higher-
                           quality Indian-language translation]
  -> voice_picker.py     librosa pitch (F0) estimate on the source audio picks a
                          male/female edge-tts voice as a stand-in for full voice
                          cloning
  -> synthesizer.py      edge-tts renders each segment, ffmpeg atempo stretches/
                          compresses it to fit the original segment's time slot
                          (clamped to 0.6x-1.7x so it stays natural), segments are
                          overlaid onto a silent track at their original offsets
  -> muxer.py            ffmpeg swaps the audio track onto the source video with
                          `-c:v copy` (no video re-encode)
  -> <name>_dubbed_en.mp4
```

### Key design decisions

- **Single-pass translation by default.** `faster-whisper` can transcribe and
  translate to English in one call (`task="translate"`), which is simpler and
  faster than pairing transcription with a separate translation model, and
  works across the languages Whisper supports out of the box. The optional
  `indictrans2` backend swaps in `ai4bharat/indictrans2-indic-en-1B` for
  Indian-language sources when higher translation quality is worth the extra
  model download and inference time; it degrades gracefully to the default
  backend if the dependencies aren't installed.
- **Timing alignment via per-segment tempo, not just start-offset.** Each
  Whisper segment carries a start/end time from the original speech. TTS
  output rarely matches that duration, so each segment is time-stretched with
  ffmpeg's `atempo` filter to fit its original slot before being placed on the
  timeline. This keeps the dub roughly lip-synced without needing full
  phoneme-level alignment.
- **Pitch-based voice selection instead of a fixed voice.** A quick `librosa.pyin`
  pitch estimate over the source audio picks a closer-matching male/female
  edge-tts voice. Full voice cloning (matching timbre exactly) is treated as a
  stretch goal — see below.
- **No re-encoding of video.** The mux step uses `-c:v copy`, only the audio
  stream is replaced, so video quality and processing time are unaffected by
  the dub.

### Not implemented (stretch goals)

Multi-speaker diarization (`pyannote.audio`) and per-speaker voice cloning
(`Coqui XTTS`) were left out to keep the core pipeline reliable first. The
current single-voice design is the natural place to plug them in:
`translator.py`'s segments would carry a speaker label, and `voice_picker.py`
would resolve a (cloned) voice per speaker instead of once per video.

## Known limitations

- One voice for the whole video (no per-speaker distinction).
- Tempo clamped to 0.6x-1.7x; segments needing more extreme stretching will
  drift slightly out of sync rather than sound distorted.
- `indictrans2` backend requires extra heavy dependencies (torch, transformers,
  IndicTransToolkit) not installed by default — see `requirements.txt`.
