FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1 \
    SAM2_BUILD_CUDA=0

RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/web && python -m venv /opt/inference
COPY apps/web/requirements.txt /tmp/web-requirements.txt
COPY apps/web/space/requirements-inference.txt /tmp/inference-requirements.txt
RUN /opt/web/bin/pip install -r /tmp/web-requirements.txt && /opt/web/bin/pip check
RUN /opt/inference/bin/pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
RUN /opt/inference/bin/pip install setuptools wheel \
    && /opt/inference/bin/pip install -r /tmp/inference-requirements.txt \
    && /opt/inference/bin/pip install --no-deps --no-build-isolation \
       git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4 \
    && /opt/inference/bin/pip check

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser apps/web /app/apps/web
ENV PYTHONPATH=/app/src VPI_WORKER_PYTHON=/opt/inference/bin/python \
    HF_HOME=/home/appuser/.cache/huggingface GRADIO_TEMP_DIR=/tmp/gradio \
    VPI_ENABLE_INFERENCE=1 VPI_ALLOW_MODEL_DOWNLOADS=0 VPI_INSPECTION_TIMEOUT=1800
USER appuser
# These checks import code/metadata only. No weights or GPU operations at build time.
RUN /opt/inference/bin/python -m visual_peer_inspector.web.worker --check \
    && /opt/web/bin/python apps/web/check_deployment.py
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD /opt/web/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"
CMD ["/opt/web/bin/python", "apps/web/app.py", "--host", "0.0.0.0", "--port", "7860"]
