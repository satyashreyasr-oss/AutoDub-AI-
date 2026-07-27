import queue
import threading
from pathlib import Path

import streamlit as st

from dubbing.pipeline import DubbingPipeline
from dubbing.utils import subscribe

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEVICES = ["auto", "cpu", "cuda"]
TRANSLATORS = ["whisper", "indictrans2"]

st.set_page_config(page_title="AutoDub", layout="wide")
st.title("AutoDub — YouTube to English Dub")

st.caption(
    "Running on a resource-limited host (e.g. Streamlit Community Cloud)? "
    "Stick to `tiny`/`base` Whisper models — `medium`/`large-v3` will likely "
    "exceed the free tier's ~1GB RAM."
)

with st.sidebar:
    url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    whisper_model = st.selectbox("Whisper model", WHISPER_MODELS, index=WHISPER_MODELS.index("tiny"))
    device = st.selectbox("Device", DEVICES, index=0)
    translator = st.selectbox("Translator backend", TRANSLATORS, index=0)
    voice = st.text_input("Force voice (optional)", placeholder="en-US-GuyNeural")
    output_dir = st.text_input("Output directory", value="output")
    keep_temp = st.checkbox("Keep temp files", value=False)
    run_clicked = st.button("Dub video", type="primary")

log_box = st.empty()
result_box = st.container()


def run_pipeline_blocking(url, output_dir, whisper_model, device, voice, translator, keep_temp):
    log_lines = []
    log_queue = queue.Queue()
    unsubscribe = subscribe(log_queue.put)
    result = {}

    def worker():
        try:
            pipeline = DubbingPipeline(
                output_dir=Path(output_dir or "output"),
                whisper_model=whisper_model,
                device=device,
                voice=(voice or "").strip() or None,
                translator_backend=translator,
                keep_temp=keep_temp,
            )
            result["path"] = pipeline.run(url)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while thread.is_alive() or not log_queue.empty():
            try:
                log_lines.append(log_queue.get(timeout=0.25))
                log_box.text_area("Progress", "\n".join(log_lines), height=400)
            except queue.Empty:
                continue
    finally:
        unsubscribe()

    return result


if run_clicked:
    url = (url or "").strip()
    if not url:
        st.error("Enter a YouTube URL first.")
    else:
        result = run_pipeline_blocking(
            url, output_dir, whisper_model, device, voice, translator, keep_temp
        )
        if "error" in result:
            st.error(f"Failed: {result['error']}")
        else:
            video_path = result["path"]
            srt_path = video_path.with_suffix(".srt")
            with result_box:
                st.success(f"Done -> {video_path}")
                st.video(str(video_path))
                if srt_path.exists():
                    st.download_button(
                        "Download subtitles (.srt)",
                        data=srt_path.read_bytes(),
                        file_name=srt_path.name,
                        mime="text/plain",
                    )
