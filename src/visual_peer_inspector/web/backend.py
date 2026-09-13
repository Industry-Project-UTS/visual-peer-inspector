"""Bounded inference subprocess; the UI environment needs no GPU dependencies."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .core import STAGES, InspectionCancelled, InspectionError
from .deployment import WorkerSettings
from .transport import read_result, write_request

ERRORS = {
    "cuda_unavailable": "Inspection needs a CUDA GPU. Check the Space's hardware setting.",
    "models_not_cached": "The model files are not cached. Enable model downloads in the deployment settings.",
    "checkpoint_integrity": "A model checkpoint failed verification. Rebuild its cache before retrying.",
    "proposal_truncated": "Object detection could not finish reliably. Try a simpler or clearer scene.",
    "out_of_memory": "The GPU ran out of memory. Try fewer views or a larger GPU.",
    "dependencies": "The inference environment is incomplete. Check the deployment build logs.",
    "policy": "The configured review policy does not match this app. Check deployment settings.",
}


def _stop(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=3)
    except ProcessLookupError:
        pass


class SubprocessBackend:
    def __init__(
        self,
        settings: WorkerSettings,
        *,
        worker_module="visual_peer_inspector.web.worker",
    ):
        self.settings = settings
        self.worker_module = worker_module
        self._gate = threading.Lock()
        self._state_lock = threading.Lock()
        self._active = {}

    def cancel(self, request_id: str) -> bool:
        with self._state_lock:
            cancel = self._active.get(request_id)
            if cancel is None:
                return False
            cancel.set()
            return True

    def inspect(self, request, progress):
        if not self._gate.acquire(blocking=False):
            raise InspectionError(
                "Another inspection is running. Try again when it finishes."
            )
        cancelled = threading.Event()
        with self._state_lock:
            self._active[request.request_id] = cancelled
        process = None
        reader = None
        temporary = None
        try:
            temporary = tempfile.TemporaryDirectory(prefix="vpi-job-")
            root = Path(temporary.name)
            write_request(root, request)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
            env["PYTHONUNBUFFERED"] = "1"
            env["VPI_ALLOW_MODEL_DOWNLOADS"] = (
                "1" if self.settings.allow_downloads else "0"
            )
            if not self.settings.allow_downloads:
                env["HF_HUB_OFFLINE"] = "1"
                env["TRANSFORMERS_OFFLINE"] = "1"
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"
            if self.settings.review_policy_path:
                env["VPI_REVIEW_POLICY"] = str(
                    Path(self.settings.review_policy_path).resolve()
                )
            else:
                env.pop("VPI_REVIEW_POLICY", None)
            process = subprocess.Popen(
                [
                    self.settings.python,
                    "-m",
                    self.worker_module,
                    "--job",
                    str(root),
                ],
                stdout=subprocess.PIPE,
                stderr=None,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                env=env,
                start_new_session=os.name == "posix",
            )
            messages = queue.Queue(maxsize=64)

            def read_messages():
                try:
                    for line in iter(lambda: process.stdout.readline(65536), ""):
                        try:
                            messages.put_nowait(json.loads(line))
                        except (ValueError, queue.Full):
                            pass
                finally:
                    process.stdout.close()

            reader = threading.Thread(target=read_messages, daemon=True)
            reader.start()
            deadline = time.monotonic() + self.settings.timeout_seconds
            error_code = None
            while process.poll() is None or reader.is_alive() or not messages.empty():
                if cancelled.is_set():
                    raise InspectionCancelled(
                        "Inspection cancelled. Your uploaded images are still available."
                    )
                if time.monotonic() >= deadline:
                    raise InspectionError(
                        "Inspection timed out. Try again; model downloads already cached can be reused."
                    )
                try:
                    message = messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                if not isinstance(message, dict):
                    continue
                if message.get("stage") in STAGES:
                    progress(message["stage"])
                if "error" in message:
                    error_code = message["error"]
            if cancelled.is_set():
                raise InspectionCancelled(
                    "Inspection cancelled. Your uploaded images are still available."
                )
            if process.returncode != 0:
                raise InspectionError(
                    ERRORS.get(
                        error_code,
                        "Inspection failed. Check the deployment logs and try again.",
                    )
                )
            return read_result(root, request)
        finally:
            # Stop the complete process group before the temporary images are removed.
            try:
                if process is not None:
                    _stop(process)
                if reader is not None:
                    reader.join(timeout=3)
                if temporary is not None:
                    temporary.cleanup()
            finally:
                with self._state_lock:
                    self._active.pop(request.request_id, None)
                self._gate.release()


def create_backend():
    return SubprocessBackend(WorkerSettings.from_environment())
