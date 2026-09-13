"""Run with: python apps/web/app.py (no GPU or datasets required)."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from visual_peer_inspector.web.core import load_backend
from visual_peer_inspector.web.ui import CSS, build_app


def main():
    parser = argparse.ArgumentParser(description="Visual Peer Inspector web app")
    parser.add_argument(
        "--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    )
    arguments = parser.parse_args()
    backend_spec = os.getenv("VPI_BACKEND")
    if backend_spec is None and os.getenv("VPI_ENABLE_INFERENCE") == "1":
        backend_spec = "visual_peer_inspector.web.backend:create_backend"
    username, password = os.getenv("VPI_AUTH_USERNAME"), os.getenv("VPI_AUTH_PASSWORD")
    if bool(username) != bool(password):
        raise ValueError(
            "Set both VPI_AUTH_USERNAME and VPI_AUTH_PASSWORD, or neither."
        )
    app = build_app(load_backend(backend_spec))
    app.launch(
        server_name=arguments.host,
        server_port=arguments.port,
        share=False,
        theme=app.vpi_theme,
        css=CSS,
        max_file_size="10mb",
        show_error=False,
        footer_links=[],
        favicon_path=str(Path(__file__).with_name("favicon.svg")),
        quiet=True,
        auth=(username, password) if username and password else None,
    )


if __name__ == "__main__":
    main()
