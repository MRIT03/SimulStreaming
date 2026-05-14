"""Bridge manager with backend-controlled performance mode.

Performance modes from GET /api/v1/settings:
- high: after call end, restart the live stack immediately while offline Whisper runs.
- moderate: stop live stack, run offline Whisper, then restart live stack.

Manual end-call endpoint:
- POST http://<bridge-host>:5003/end-call
- GET  http://<bridge-host>:5003/status
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

FASTAPI_STREAM_HOST = os.getenv("FASTAPI_STREAM_HOST", "127.0.0.1")
FASTAPI_STREAM_PORT = int(os.getenv("FASTAPI_STREAM_PORT", "5001"))

FASTAPI_OFFLINE_HOST = os.getenv("FASTAPI_OFFLINE_HOST", "127.0.0.1")
FASTAPI_OFFLINE_PORT = int(os.getenv("FASTAPI_OFFLINE_PORT", "5002"))

WHISPER_HOST = os.getenv("WHISPER_HOST", "127.0.0.1")
WHISPER_PORT = int(os.getenv("WHISPER_PORT", "43002"))

AUDIO_PROXY_HOST = os.getenv("AUDIO_PROXY_HOST", "127.0.0.1")
AUDIO_PROXY_PORT = int(os.getenv("AUDIO_PROXY_PORT", "43001"))

# Bind to 0.0.0.0 so Streamlit can reach it from another process/container.
CONTROL_SERVER_HOST = os.getenv("CONTROL_SERVER_HOST", "0.0.0.0")
CONTROL_SERVER_PORT = int(os.getenv("CONTROL_SERVER_PORT", "5003"))

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./audio")).resolve()
AUDIO_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

CALL_END_TIMEOUT_SECONDS = float(os.getenv("CALL_END_TIMEOUT_SECONDS", "8"))
SILENCE_DURATION_SECONDS = float(os.getenv("SILENCE_DURATION_SECONDS", "10"))
SILENCE_THRESHOLD_DB = os.getenv("SILENCE_THRESHOLD_DB", "-35dB")

WHISPER_SERVER_CMD = [
    "python",
    ".\\simulstreaming_whisper_server.py",
    "--host", "localhost",
    "--port", str(WHISPER_PORT),
    "--language", "ar",
]

FFMPEG_CMD = [
    "ffmpeg",
    "-re",
    "-f", "dshow",
    "-rtbufsize", "50M",
    "-i", os.getenv("FFMPEG_AUDIO_INPUT", "audio=Mikrofonarray (Realtek(R) Audio)"),
    "-ac", "1",
    "-ar", "16000",
    "-af", f"silencedetect=noise={SILENCE_THRESHOLD_DB}:d={SILENCE_DURATION_SECONDS}",
    "-acodec", "pcm_s16le",
    "-f", "s16le",
    f"tcp://{AUDIO_PROXY_HOST}:{AUDIO_PROXY_PORT}",
]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

stop_event = threading.Event()
_call_active = threading.Event()
_cycle_restart_event = threading.Event()
_offline_done = threading.Event()
_offline_done.set()

_call_id_lock = threading.Lock()
_current_call_id: str | None = None

_last_transcript_lock = threading.Lock()
_last_transcript_time = 0.0

_call_end_lock = threading.Lock()
_call_end_in_progress = False

_wav_lock = threading.Lock()
_wav_file = None
_wav_path: Path | None = None

_proc_lock = threading.Lock()
_whisper_proc: subprocess.Popen | None = None
_ffmpeg_proc: subprocess.Popen | None = None

_latest_mode_lock = threading.Lock()
_latest_performance_mode = "moderate"

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _fetch_performance_mode() -> str:
    global _latest_performance_mode
    try:
        with request.urlopen(f"{BACKEND_API_URL}/api/v1/settings", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        mode = data.get("performance", "moderate")
        if mode not in {"high", "moderate"}:
            mode = "moderate"
    except Exception as exc:
        print(f"[Settings] Could not fetch settings, using cached/default mode: {exc}", flush=True)
        with _latest_mode_lock:
            return _latest_performance_mode

    with _latest_mode_lock:
        _latest_performance_mode = mode
    print(f"[Settings] Performance mode: {mode}", flush=True)
    return mode


def _get_cached_mode() -> str:
    with _latest_mode_lock:
        return _latest_performance_mode

# ---------------------------------------------------------------------------
# Socket helpers
# ---------------------------------------------------------------------------


def _connect_socket(host: str, port: int, label: str) -> socket.socket:
    while not stop_event.is_set():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            print(f"[Bridge] Connected to {label} at {host}:{port}", flush=True)
            return sock
        except OSError as exc:
            print(f"[Bridge] Waiting for {label} at {host}:{port}: {exc}", flush=True)
            time.sleep(1)
    raise RuntimeError("Bridge stopped before socket connection")


def _send_json(sock_ref: list[socket.socket], host: str, port: int, label: str, payload: dict) -> bool:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    for _ in range(2):
        try:
            sock_ref[0].sendall(data)
            print(f"[Bridge] Sent to {label}: {payload.get('type')}", flush=True)
            return True
        except OSError as exc:
            print(f"[Bridge] Send failed to {label}: {exc}", flush=True)
            try:
                sock_ref[0].close()
            except Exception:
                pass
            sock_ref[0] = _connect_socket(host, port, label)
    return False

# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _stop_process(proc: subprocess.Popen | None, label: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"[{label}] Stopping", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"[{label}] Killing", flush=True)
        proc.kill()


def _set_processes(whisper: subprocess.Popen | None = None, ffmpeg: subprocess.Popen | None = None) -> None:
    global _whisper_proc, _ffmpeg_proc
    with _proc_lock:
        if whisper is not None:
            _whisper_proc = whisper
        if ffmpeg is not None:
            _ffmpeg_proc = ffmpeg


def _get_processes() -> tuple[subprocess.Popen | None, subprocess.Popen | None]:
    with _proc_lock:
        return _whisper_proc, _ffmpeg_proc


def _clear_processes() -> None:
    global _whisper_proc, _ffmpeg_proc
    with _proc_lock:
        _whisper_proc = None
        _ffmpeg_proc = None

# ---------------------------------------------------------------------------
# Call / WAV helpers
# ---------------------------------------------------------------------------


def _start_new_call() -> str:
    global _current_call_id, _last_transcript_time
    call_id = str(uuid.uuid4())
    with _call_id_lock:
        _current_call_id = call_id
    with _last_transcript_lock:
        _last_transcript_time = time.monotonic()
    _call_active.set()
    print(f"[Call] New call started: {call_id}", flush=True)
    return call_id


def _get_call_id() -> str | None:
    with _call_id_lock:
        return _current_call_id


def _clear_call_id() -> str | None:
    global _current_call_id
    with _call_id_lock:
        call_id = _current_call_id
        _current_call_id = None
    _call_active.clear()
    return call_id


def _open_wav() -> None:
    global _wav_file, _wav_path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = AUDIO_DIR / f"call_{timestamp}.wav"
    wav = wave.open(str(path), "wb")
    wav.setnchannels(CHANNELS)
    wav.setsampwidth(SAMPLE_WIDTH)
    wav.setframerate(SAMPLE_RATE)
    _wav_file = wav
    _wav_path = path
    print(f"[Audio] Recording to {path}", flush=True)


def _write_audio(data: bytes) -> None:
    with _wav_lock:
        if _wav_file is None:
            _open_wav()
        _wav_file.writeframes(data)


def _finalize_wav() -> Path | None:
    global _wav_file, _wav_path
    with _wav_lock:
        if _wav_file is None:
            return None
        path = _wav_path
        _wav_file.close()
        _wav_file = None
        _wav_path = None
    print(f"[Audio] Finalized WAV: {path}", flush=True)
    return path

# ---------------------------------------------------------------------------
# Offline Whisper
# ---------------------------------------------------------------------------


def _extract_text(output: str) -> str | None:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = data.get("text", "").strip()
        if text:
            return text
    return None


def _run_offline_whisper(wav_path: Path | None, call_id: str, offline_sock_ref: list[socket.socket]) -> None:
    if wav_path is None or not wav_path.exists():
        print("[Offline] WAV not found, skipping", flush=True)
        return

    print(f"[Offline] Processing {wav_path.name} for call {call_id[:8]}", flush=True)
    cmd = ["python", ".\\simulstreaming_whisper.py", "--lan", "ar", str(wav_path)]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        text = _extract_text(result.stdout)
        if not text:
            print("[Offline] No text found in output", flush=True)
            print(result.stdout[-1500:], flush=True)
            return

        payload = {
            "type": "offline_transcript",
            "call_id": call_id,
            "text": text,
            "audio_path": str(wav_path.resolve()),
            "segment_id": wav_path.stem,
        }
        _send_json(offline_sock_ref, FASTAPI_OFFLINE_HOST, FASTAPI_OFFLINE_PORT, "FastAPI offline 5002", payload)
    except Exception as exc:
        print(f"[Offline] Error: {exc}", flush=True)

# ---------------------------------------------------------------------------
# Call-end handling
# ---------------------------------------------------------------------------


def _trigger_call_end(reason: str, offline_sock_ref: list[socket.socket]) -> None:
    global _call_end_in_progress

    with _call_end_lock:
        if _call_end_in_progress:
            return
        if not _call_active.is_set():
            print(f"[Call] End requested but no active call. Reason: {reason}", flush=True)
            return
        _call_end_in_progress = True
        _offline_done.clear()

    mode = _fetch_performance_mode()
    print(f"[Call] Call ended: {reason} | mode={mode}", flush=True)

    call_id = _clear_call_id()
    whisper_proc, ffmpeg_proc = _get_processes()

    # Always stop FFmpeg to close the current audio stream and finalize the WAV.
    _stop_process(ffmpeg_proc, "FFmpeg")

    if mode == "moderate":
        # Limited hardware: free streaming Whisper resources before offline model runs.
        _stop_process(whisper_proc, "Whisper")

    wav_path = _finalize_wav()

    def offline_worker() -> None:
        global _call_end_in_progress
        try:
            if call_id:
                _run_offline_whisper(wav_path, call_id, offline_sock_ref)
        finally:
            with _call_end_lock:
                _call_end_in_progress = False
            _offline_done.set()
            print("[Bridge] Offline stage complete", flush=True)

    threading.Thread(target=offline_worker, daemon=True).start()

    # Restart policy:
    # high: live stack may restart immediately while offline thread runs.
    # moderate: supervisor waits for offline_done before restart.
    _cycle_restart_event.set()

# ---------------------------------------------------------------------------
# Watchdogs / control server
# ---------------------------------------------------------------------------


def _inactivity_watchdog(offline_sock_ref: list[socket.socket]) -> None:
    while not stop_event.is_set():
        time.sleep(1)
        if not _call_active.is_set():
            continue
        with _last_transcript_lock:
            elapsed = time.monotonic() - _last_transcript_time
        if elapsed >= CALL_END_TIMEOUT_SECONDS:
            _trigger_call_end(f"inactivity {elapsed:.1f}s", offline_sock_ref)


def _ffmpeg_stderr_reader(proc: subprocess.Popen, offline_sock_ref: list[socket.socket]) -> None:
    if proc.stderr is None:
        return
    silence_re = re.compile(r"silence_start")
    for line in proc.stderr:
        if stop_event.is_set() or _cycle_restart_event.is_set():
            break
        if silence_re.search(line) and _call_active.is_set():
            _trigger_call_end("ffmpeg silence_start", offline_sock_ref)
            break


class ControlHandler(BaseHTTPRequestHandler):
    offline_sock_ref: list[socket.socket] | None = None

    def do_GET(self):
        if self.path == "/status":
            self._respond(200, {
                "ok": True,
                "active_call": _call_active.is_set(),
                "performance": _get_cached_mode(),
            })
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/end-call":
            _trigger_call_end("manual button", self.__class__.offline_sock_ref)
            self._respond(200, {"ok": True, "message": "end-call accepted"})
        else:
            self._respond(404, {"error": "not found"})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def _respond(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def _start_control_server(offline_sock_ref: list[socket.socket]) -> None:
    ControlHandler.offline_sock_ref = offline_sock_ref
    server = ThreadingHTTPServer((CONTROL_SERVER_HOST, CONTROL_SERVER_PORT), ControlHandler)
    print(f"[Control] Listening on {CONTROL_SERVER_HOST}:{CONTROL_SERVER_PORT}", flush=True)
    server.serve_forever()

# ---------------------------------------------------------------------------
# Audio + live stack
# ---------------------------------------------------------------------------


def _audio_proxy(cycle_stop_event: threading.Event) -> None:
    print(f"[Audio Proxy] Listening on {AUDIO_PROXY_HOST}:{AUDIO_PROXY_PORT}", flush=True)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.settimeout(1)
    ffmpeg_conn = None
    whisper_conn = None

    try:
        server_sock.bind((AUDIO_PROXY_HOST, AUDIO_PROXY_PORT))
        server_sock.listen(1)
        while not stop_event.is_set() and not cycle_stop_event.is_set():
            try:
                ffmpeg_conn, addr = server_sock.accept()
                print(f"[Audio Proxy] FFmpeg connected from {addr}", flush=True)
                break
            except socket.timeout:
                continue
        if ffmpeg_conn is None:
            return

        whisper_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        whisper_conn.connect((WHISPER_HOST, WHISPER_PORT))
        print(f"[Audio Proxy] Connected to Whisper at {WHISPER_HOST}:{WHISPER_PORT}", flush=True)

        while not stop_event.is_set() and not cycle_stop_event.is_set():
            data = ffmpeg_conn.recv(4096)
            if not data:
                break
            _write_audio(data)
            try:
                whisper_conn.sendall(data)
            except OSError:
                break
    except Exception as exc:
        print(f"[Audio Proxy] Error: {exc}", flush=True)
    finally:
        for conn in (ffmpeg_conn, whisper_conn, server_sock):
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
        print("[Audio Proxy] Closed", flush=True)


def _start_whisper() -> subprocess.Popen:
    proc = subprocess.Popen(
        WHISPER_SERVER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    _set_processes(whisper=proc)
    print("[Whisper] Starting", flush=True)
    return proc


def _start_ffmpeg(offline_sock_ref: list[socket.socket]) -> subprocess.Popen:
    proc = subprocess.Popen(
        FFMPEG_CMD,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    _set_processes(ffmpeg=proc)
    threading.Thread(target=_ffmpeg_stderr_reader, args=(proc, offline_sock_ref), daemon=True).start()
    print("[FFmpeg] Running", flush=True)
    return proc


def _consume_whisper_stdout(whisper_proc: subprocess.Popen, stream_sock_ref: list[socket.socket], offline_sock_ref: list[socket.socket], cycle_stop_event: threading.Event) -> None:
    ffmpeg_started = False
    proxy_started = False
    if whisper_proc.stdout is None:
        return

    for line in whisper_proc.stdout:
        if stop_event.is_set() or cycle_stop_event.is_set() or _cycle_restart_event.is_set():
            break
        clean = line.strip()
        if not clean:
            continue
        print(f"[Whisper] {clean}", flush=True)

        if "Listening on" in clean and str(WHISPER_PORT) in clean:
            print("[Whisper] Ready", flush=True)
            if not proxy_started:
                threading.Thread(target=_audio_proxy, args=(cycle_stop_event,), daemon=True).start()
                proxy_started = True
                time.sleep(0.5)
            if not ffmpeg_started:
                _start_ffmpeg(offline_sock_ref)
                ffmpeg_started = True
            continue

        if not clean.startswith("{"):
            continue
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            continue
        text = data.get("text", "").strip()
        if not text:
            continue

        call_id = _get_call_id() or _start_new_call()
        global _last_transcript_time
        with _last_transcript_lock:
            _last_transcript_time = time.monotonic()

        payload = {
            "type": "transcript",
            "call_id": call_id,
            "text": text,
            "start": data.get("start"),
            "end": data.get("end"),
            "is_final": True,
            "emission_time": data.get("emission_time"),
        }
        _send_json(stream_sock_ref, FASTAPI_STREAM_HOST, FASTAPI_STREAM_PORT, "FastAPI live 5001", payload)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _fetch_performance_mode()

    stream_sock_ref = [_connect_socket(FASTAPI_STREAM_HOST, FASTAPI_STREAM_PORT, "FastAPI live 5001")]
    offline_sock_ref = [_connect_socket(FASTAPI_OFFLINE_HOST, FASTAPI_OFFLINE_PORT, "FastAPI offline 5002")]

    threading.Thread(target=_inactivity_watchdog, args=(offline_sock_ref,), daemon=True).start()
    threading.Thread(target=_start_control_server, args=(offline_sock_ref,), daemon=True).start()

    try:
        while not stop_event.is_set():
            mode = _fetch_performance_mode()
            if mode == "moderate":
                _offline_done.wait()

            _cycle_restart_event.clear()
            cycle_stop_event = threading.Event()
            whisper_proc = _start_whisper()

            try:
                _consume_whisper_stdout(whisper_proc, stream_sock_ref, offline_sock_ref, cycle_stop_event)
            finally:
                cycle_stop_event.set()
                whisper_proc, ffmpeg_proc = _get_processes()
                _stop_process(ffmpeg_proc, "FFmpeg")
                _stop_process(whisper_proc, "Whisper")
                _clear_processes()

            if not stop_event.is_set():
                mode = _fetch_performance_mode()
                if mode == "moderate":
                    _offline_done.wait()
                print("[Bridge] Restarting live stack", flush=True)
                time.sleep(1)

    except KeyboardInterrupt:
        print("[Bridge] Interrupted", flush=True)
    finally:
        stop_event.set()
        whisper_proc, ffmpeg_proc = _get_processes()
        _stop_process(ffmpeg_proc, "FFmpeg")
        _stop_process(whisper_proc, "Whisper")
        with _wav_lock:
            global _wav_file
            if _wav_file is not None:
                _wav_file.close()
                _wav_file = None
        for sock_ref in (stream_sock_ref, offline_sock_ref):
            try:
                sock_ref[0].close()
            except Exception:
                pass
        print("[Bridge] Done", flush=True)


if __name__ == "__main__":
    main()
