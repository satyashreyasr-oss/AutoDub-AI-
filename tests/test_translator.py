from unittest.mock import patch

from dubbing.transcriber import Segment, TranscriptionResult
from dubbing.translator import get_english_segments


def _result(language: str, text: str = "hola") -> TranscriptionResult:
    return TranscriptionResult(language=language, segments=[Segment(start=0.0, end=1.0, text=text)])


def test_default_backend_uses_whisper_translate_directly():
    with patch("dubbing.translator.run_whisper") as run_whisper:
        run_whisper.return_value = _result("es")
        get_english_segments("audio.wav", "small", "cpu", "whisper")

    run_whisper.assert_called_once_with("audio.wav", "small", "cpu", task="translate")


def test_indictrans2_backend_falls_back_to_whisper_for_non_indic_language():
    with patch("dubbing.translator.run_whisper") as run_whisper, \
         patch("dubbing.translator._translate_with_indictrans2") as translate_indic:
        run_whisper.side_effect = [_result("es"), _result("es")]

        get_english_segments("audio.wav", "small", "cpu", "indictrans2")

        translate_indic.assert_not_called()
        run_whisper.assert_any_call("audio.wav", "small", "cpu", task="transcribe")
        run_whisper.assert_any_call("audio.wav", "small", "cpu", task="translate")


def test_indictrans2_backend_used_for_indic_language():
    source = _result("hi", text="नमस्ते")
    translated = TranscriptionResult(
        language="hi", segments=[Segment(start=0.0, end=1.0, text="Hello")]
    )
    with patch("dubbing.translator.run_whisper") as run_whisper, \
         patch("dubbing.translator._translate_with_indictrans2") as translate_indic:
        run_whisper.return_value = source
        translate_indic.return_value = translated

        result = get_english_segments("audio.wav", "small", "cpu", "indictrans2")

        translate_indic.assert_called_once_with(source)
        run_whisper.assert_called_once_with("audio.wav", "small", "cpu", task="transcribe")
        assert result is translated


def test_indictrans2_failure_falls_back_to_whisper_translate():
    source = _result("hi")
    with patch("dubbing.translator.run_whisper") as run_whisper, \
         patch("dubbing.translator._translate_with_indictrans2") as translate_indic:
        run_whisper.side_effect = [source, _result("hi")]
        translate_indic.side_effect = RuntimeError("model download failed")

        get_english_segments("audio.wav", "small", "cpu", "indictrans2")

        run_whisper.assert_any_call("audio.wav", "small", "cpu", task="translate")
