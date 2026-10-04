#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"
export VPI_WORKER_PYTHON="$PWD/.runtime/inference/bin/python"
export VPI_DINO_PYTHON="$PWD/.runtime/dino/bin/python"
export HF_HOME="$PWD/.runtime/model-cache"
export GRADIO_TEMP_DIR="$PWD/.runtime/gradio-temp"
export GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1
export VPI_ENABLE_INFERENCE="${VPI_ENABLE_INFERENCE:-1}"
export VPI_ALLOW_MODEL_DOWNLOADS=0 VPI_INSPECTION_TIMEOUT=1800
if [[ -z "${GRADIO_ROOT_PATH:-}" && -f .runtime/studio-app-url.txt ]]; then
  GRADIO_ROOT_PATH="$(.runtime/web/bin/python - <<'PY'
from pathlib import Path
from urllib.parse import urlsplit
url = Path('.runtime/studio-app-url.txt').read_text().strip().rstrip('/')
parts = urlsplit(url)
if (parts.scheme != 'https' or not parts.hostname or parts.username or
        parts.password or parts.query or parts.fragment or
        parts.path != '/jupyterlab/default/proxy/7860'):
    raise SystemExit('Invalid HTTPS Studio app URL in .runtime/studio-app-url.txt')
print(url)
PY
)"
fi
export GRADIO_ROOT_PATH="${GRADIO_ROOT_PATH:-/jupyterlab/default/proxy/7860}"
mkdir -p "$GRADIO_TEMP_DIR" logs
exec .runtime/web/bin/python apps/web/app.py --host 127.0.0.1 --port 7860
