#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=3 SAM2_BUILD_CUDA=0
export GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PWD/src"
mkdir -p .runtime logs
exec 9>.runtime/setup.lock
flock -n 9 || { echo 'Another setup process is running.'; exit 1; }
trap 'rc=$?; printf "setup_failed exit=%s line=%s\n" "$rc" "$LINENO" | tee logs/setup-status.txt; exit "$rc"' ERR
python - <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'
for row in json.loads(Path('package-manifest.json').read_text()):
    path = Path(row['path'])
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], f'Package mismatch: {path}'
free = shutil.disk_usage('.').free / 1024**3
assert free >= 25, f'Need at least 25 GiB free for first install; have {free:.1f}'
print(f'Package verified; {free:.1f} GiB free', flush=True)
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
for envname in web dino inference; do
    if [[ ! -f ".runtime/$envname/bin/python" ]]; then
        python -m venv ".runtime/$envname"
    fi
done
echo 'Installing isolated web environment'
.runtime/web/bin/python -m pip install -r apps/web/requirements.txt
.runtime/web/bin/python -m pip check
.runtime/web/bin/python apps/web/check_deployment.py | tee logs/web-preflight.json
echo 'web_ready' > logs/setup-status.txt
echo 'Installing exact CPU DINO environment'
.runtime/dino/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.runtime/dino/bin/python -m pip install -r apps/web/space/requirements-dino.txt
.runtime/dino/bin/python -m pip check
.runtime/dino/bin/python - <<'PY'
from visual_peer_inspector.web.encoder_worker import compatible_runtime
assert compatible_runtime(), 'DINO runtime differs from the frozen contract'
print('DINO runtime verified', flush=True)
PY
echo 'dino_ready' > logs/setup-status.txt
echo 'Installing CUDA detector and SAM environment'
.runtime/inference/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.runtime/inference/bin/python -m pip install setuptools wheel
.runtime/inference/bin/python -m pip install -r apps/web/space/requirements-inference.txt
.runtime/inference/bin/python -m pip install --no-deps --no-build-isolation git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4
.runtime/inference/bin/python apps/web/sagemaker/check_environment.py
.runtime/inference/bin/python - <<'PY'
import torch
from sam2.build_sam import build_sam2
from visual_peer_inspector.models.sam2 import verify_installed_sam2_revision
assert torch.cuda.is_available(), 'CUDA unavailable in inference environment'
verify_installed_sam2_revision('2b90b9f5ceec907a1c18123530e92e794ad901a4')
print('CUDA and pinned SAM source verified:', torch.cuda.get_device_name(0), flush=True)
PY
.runtime/inference/bin/python -m visual_peer_inspector.web.worker --check | tee logs/worker-preflight.json
df -h .
echo 'environments_ready' | tee logs/setup-status.txt
