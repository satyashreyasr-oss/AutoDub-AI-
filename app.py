import queue
import threading
from pathlib import Path

import gradio as gr

from dubbing.pipeline import DubbingPipeline
from dubbing.utils import subscribe

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEVICES = ["auto", "cpu", "cuda"]
TRANSLATORS = ["whisper", "indictrans2"]


def run_pipeline(url, output_dir, whisper_model, device, voice, translator, keep_temp):
    url = (url or "").strip()
    if not url:
        yield "Enter a YouTube URL first.", None, None
        return

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
                yield "\n".join(log_lines), None, None
            except queue.Empty:
                continue
    finally:
        unsubscribe()

    if "error" in result:
        log_lines.append(f"ERROR: {result['error']}")
        yield "\n".join(log_lines), None, None
        return

    video_path = result["path"]
    srt_path = video_path.with_suffix(".srt")
    yield (
        "\n".join(log_lines),
        str(video_path),
        str(srt_path) if srt_path.exists() else None,
    )


with gr.Blocks(title="AutoDub") as demo:
    gr.Markdown("# AutoDub — YouTube to English Dub")
    with gr.Row():
        with gr.Column(scale=2):
            url = gr.Textbox(
                label="YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
            )
            with gr.Row():
                whisper_model = gr.Dropdown(WHISPER_MODELS, value="small", label="Whisper model")
                device = gr.Dropdown(DEVICES, value="auto", label="Device")
            with gr.Row():
                translator = gr.Dropdown(TRANSLATORS, value="whisper", label="Translator backend")
                voice = gr.Textbox(label="Force voice (optional)", placeholder="en-US-GuyNeural")
            with gr.Row():
                output_dir = gr.Textbox(value="output", label="Output directory")
                keep_temp = gr.Checkbox(label="Keep temp files", value=False)
            run_btn = gr.Button("Dub video", variant="primary")
        with gr.Column(scale=3):
            log_box = gr.Textbox(label="Progress", lines=20, autoscroll=True, interactive=False)
            video_out = gr.Video(label="Dubbed video")
            srt_out = gr.File(label="Subtitles (.srt)")

    run_btn.click(
        run_pipeline,
        inputs=[url, output_dir, whisper_model, device, voice, translator, keep_temp],
        outputs=[log_box, video_out, srt_out],
    )


if __name__ == "__main__":
    demo.queue().launch()
