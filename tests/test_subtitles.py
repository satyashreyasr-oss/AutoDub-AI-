from dubbing.subtitles import format_srt_timestamp, write_srt
from dubbing.transcriber import Segment


def test_format_srt_timestamp_zero():
    assert format_srt_timestamp(0) == "00:00:00,000"


def test_format_srt_timestamp_rolls_over_hours_minutes_seconds():
    assert format_srt_timestamp(3661.234) == "01:01:01,234"


def test_format_srt_timestamp_rounds_milliseconds():
    assert format_srt_timestamp(1.9996) == "00:00:02,000"


def test_write_srt_writes_sequential_numbered_cues(tmp_path):
    segments = [
        Segment(start=0.0, end=1.5, text="Hello there."),
        Segment(start=1.5, end=3.0, text="General Kenobi."),
    ]
    out_path = tmp_path / "out.srt"

    write_srt(segments, out_path)
    content = out_path.read_text(encoding="utf-8")

    assert content == (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "Hello there.\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 00:00:03,000\n"
        "General Kenobi.\n"
    )


def test_write_srt_skips_empty_segments_without_gapping_numbering(tmp_path):
    segments = [
        Segment(start=0.0, end=1.0, text="First."),
        Segment(start=1.0, end=2.0, text=""),
        Segment(start=2.0, end=3.0, text="Third."),
    ]
    out_path = tmp_path / "out.srt"

    write_srt(segments, out_path)
    content = out_path.read_text(encoding="utf-8")

    assert "1\n00:00:00,000 --> 00:00:01,000\nFirst." in content
    assert "2\n00:00:02,000 --> 00:00:03,000\nThird." in content
    assert "3\n" not in content
