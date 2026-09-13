"""Bounded notebook subprocesses, recovery copies and optional post-run tracking."""

import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time


def run_bounded(command, log_path, *, idle_seconds=900, total_seconds=18000):
    """Keep progress visible and stop a silent worker without deleting checkpoints."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = last_output = time.monotonic()
    lines = queue.Queue()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    def collect():
        for line in process.stdout:
            lines.put(line)
    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    timed_out = False
    with log_path.open("a") as stream:
        while process.poll() is None or reader.is_alive() or not lines.empty():
            try:
                line = lines.get(timeout=1)
                last_output = time.monotonic()
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            except queue.Empty:
                pass
            now = time.monotonic()
            if process.poll() is None and (now - last_output > idle_seconds or now - started > total_seconds):
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
    process.wait()
    if timed_out:
        raise TimeoutError("Worker stopped after the declared time limit. Retain outputs and inspect the last checkpoint.")
    if process.returncode:
        raise RuntimeError(f"Stage failed with exit code {process.returncode}. Inspect the local log before restarting.")


def restore_recovery(source, destination):
    """Copy a downloaded working directory without overwriting another run."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("Recovery destination must be empty")
    if not (source / "run_binding.json").is_file():
        raise ValueError("Recovery input has no run binding")
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Recovery directories cannot contain symlinks")
    shutil.copytree(source, destination)
    # The prediction stage checks source, environment, input and scene receipts before reuse.


def tracking_credentials():
    """Read configured keys without printing or writing them to artifacts."""
    keys = ("CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")
    if not all(os.environ.get(key) for key in keys):
        try:
            from kaggle_secrets import UserSecretsClient
            secrets = UserSecretsClient()
            for key in keys:
                if not os.environ.get(key):
                    os.environ[key] = secrets.get_secret(key)
        except Exception:
            return False
    os.environ.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
    os.environ.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
    os.environ.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")
    return all(os.environ.get(key) for key in keys)


def upload_tracking(source, evidence, notebook):
    """Tracking runs after local evidence is saved, with a strict timeout."""
    evidence = Path(evidence)
    receipt = evidence / "tracking_receipt.json"
    if not tracking_credentials():
        receipt.write_text(json.dumps({"status": "not_uploaded", "reason": "configured secrets unavailable"}))
        print("ClearML was not uploaded. Scientific outputs are preserved; retry tracking separately.")
        return
    command = [sys.executable, str(Path(source) / "scripts/log_scoring_comparison.py"), "--evidence", str(evidence), "--notebook", notebook]
    try:
        subprocess.run(command, check=True, timeout=180)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        receipt.write_text(json.dumps({"status": "upload_not_verified", "reason": "bounded upload did not complete"}))
        print("ClearML upload was not verified. Do not rerun inference to retry tracking.")
