from pathlib import Path

import numpy as np

from .utils import log

MALE_VOICE = "en-US-GuyNeural"
FEMALE_VOICE = "en-US-JennyNeural"
DEFAULT_VOICE = FEMALE_VOICE

# Rough male/female F0 boundary in Hz.
PITCH_THRESHOLD_HZ = 165.0


def pick_voice(audio_path: Path, forced_voice: str | None) -> str:
    if forced_voice:
        log("voice", f"Using forced voice: {forced_voice}")
        return forced_voice

    try:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=16000, mono=True, duration=180)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6")
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) == 0:
            log("voice", "Could not detect pitch; defaulting to a neutral voice")
            return DEFAULT_VOICE

        mean_pitch = float(np.mean(voiced_f0))
        voice = MALE_VOICE if mean_pitch < PITCH_THRESHOLD_HZ else FEMALE_VOICE
        log("voice", f"Estimated mean pitch {mean_pitch:.0f} Hz -> selecting {voice}")
        return voice
    except Exception as exc:
        log("voice", f"Pitch detection failed ({exc}); defaulting to a neutral voice")
        return DEFAULT_VOICE
