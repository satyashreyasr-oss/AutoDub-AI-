import subprocess
import time
from typing import Callable

_subscribers: list[Callable[[str], None]] = []


def subscribe(callback: Callable[[str], None]) -> Callable[[], None]:
    """Register a callback to receive every future log line as it's emitted.

    Returns an unsubscribe function. Used by the UI to stream pipeline
    progress without every module having to know about it.
    """
    _subscribers.append(callback)
    return lambda: _subscribers.remove(callback)


def log(step: str, message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] [{step}] {message}"
    print(line, flush=True)
    for callback in list(_subscribers):
        callback(line)


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
