"""Start the private Studio app once, or inspect/stop its owned process."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]


class Service:
    def __init__(self, root: Path, port: int = 7860):
        self.root = root.resolve()
        self.port = port
        self.logs = self.root / "logs"
        self.logs.mkdir(exist_ok=True)
        self.pid_file = self.logs / "app.pid"

    def owned_pid(self):
        try:
            pid = int(self.pid_file.read_text().strip())
            if pid <= 1:
                return None
            process = Path("/proc") / str(pid)
            args = process.joinpath("cmdline").read_bytes().split(b"\0")
            cwd = process.joinpath("cwd").resolve(strict=True)
            expected = {b"apps/web/app.py", str(self.root / "apps/web/app.py").encode()}
            interpreters = {b".runtime/web/bin/python", str(self.root / ".runtime/web/bin/python").encode()}
            return pid if (cwd == self.root and len(args) > 1
                           and args[0] in interpreters and args[1] in expected) else None
        except (OSError, ValueError):
            return None

    def healthy(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=2) as response:
                return response.status == 200
        except OSError:
            return False

    def status(self):
        pid = self.owned_pid()
        return {"running": pid is not None, "pid": pid,
                "http_ready": self.healthy() if pid else False,
                "model_inference_verified": False}

    def start(self, timeout=60):
        pid = self.owned_pid()
        if pid:
            return {**self.status(), "already_running": True}
        # A foreign listener must never be replaced or killed.
        with socket.socket() as sock:
            # The previous owned server can leave closed connections in TIME_WAIT.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.port))
        launch = self.root / "apps/web/sagemaker/launch_sagemaker.sh"
        if not launch.is_file():
            raise RuntimeError("SageMaker launch script is missing.")
        with (self.logs / "app.log").open("ab") as log:
            process = subprocess.Popen(["bash", str(launch)], cwd=self.root,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        temporary = self.pid_file.with_suffix(".tmp")
        temporary.write_text(str(process.pid) + "\n")
        temporary.replace(self.pid_file)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("App exited during startup. See logs/app.log.")
            if self.owned_pid() and self.healthy():
                return {**self.status(), "already_running": False}
            time.sleep(0.25)
        # Keep the PID and log for diagnosis; never report a timeout as success.
        raise RuntimeError("App did not become ready within the startup timeout. See logs/app.log.")

    def stop(self, timeout=10):
        pid = self.owned_pid()
        if not pid:
            return {"stopped": False, "reason": "No owned app process."}
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        for child in children:
            with contextlib.suppress(OSError):
                command = Path(f"/proc/{child}/cmdline").read_bytes()
                if b"visual_peer_inspector.web.worker" in command:
                    raise RuntimeError("An inspection is running. Cancel it in the app before stopping.")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while self.owned_pid() and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.owned_pid():
            raise RuntimeError("App did not stop; its process was not forcibly killed.")
        self.pid_file.unlink(missing_ok=True)
        return {"stopped": True, "pid": pid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    args = parser.parse_args()
    if not Path("/proc").is_dir():
        parser.error("This helper runs inside the Linux SageMaker Space.")
    import fcntl
    service = Service(ROOT)
    with (service.logs / "service.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(getattr(service, args.action)(), indent=2))


if __name__ == "__main__":
    main()
