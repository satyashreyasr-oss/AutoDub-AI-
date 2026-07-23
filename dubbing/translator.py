from pathlib import Path

from .transcriber import Segment, TranscriptionResult, run_whisper
from .utils import log

INDIC_LANGUAGES = {"hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"}

INDIC_LANG_CODES = {
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
}


def get_english_segments(audio_path: Path, model_size: str, device: str, backend: str) -> TranscriptionResult:
    if backend == "indictrans2":
        source = run_whisper(audio_path, model_size, device, task="transcribe")
        if source.language not in INDIC_LANGUAGES:
            log(
                "translate",
                f"Source language '{source.language}' is not an Indic language; using Whisper translation instead",
            )
            return run_whisper(audio_path, model_size, device, task="translate")
        try:
            return _translate_with_indictrans2(source)
        except Exception as exc:
            log("translate", f"IndicTrans2 unavailable ({exc}); falling back to Whisper translation")
            return run_whisper(audio_path, model_size, device, task="translate")

    return run_whisper(audio_path, model_size, device, task="translate")


def _translate_with_indictrans2(source: TranscriptionResult) -> TranscriptionResult:
    import torch
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "ai4bharat/indictrans2-indic-en-1B"
    log("translate", f"Loading IndicTrans2 model '{model_name}'")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    processor = IndicProcessor(inference=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    src_lang = INDIC_LANG_CODES.get(source.language, "hin_Deva")
    texts = [seg.text for seg in source.segments]

    batch = processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang="eng_Latn")
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_length=256, num_beams=5)

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    translations = processor.postprocess_batch(decoded, lang="eng_Latn")

    translated_segments = [
        Segment(start=seg.start, end=seg.end, text=text.strip())
        for seg, text in zip(source.segments, translations)
    ]
    log("translate", f"Translated {len(translated_segments)} segments with IndicTrans2")
    return TranscriptionResult(language=source.language, segments=translated_segments)
