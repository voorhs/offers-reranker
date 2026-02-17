# Build stage
FROM python:3.12-slim AS builder

WORKDIR /build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY predict/pyproject.toml /build/
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -e ".[export]"

COPY models/rubert-tiny-reranker /build/model
COPY predict/export_onnx.py /build/

RUN python export_onnx.py --model-path /build/model --output-dir /build/onnx_model

# ONNX Runtime
FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY predict/pyproject.toml /app/
RUN uv pip install --system --no-cache -e ".[runtime]"

COPY --from=builder /build/onnx_model /model

COPY predict/predict_onnx.py ./predict.py

ENTRYPOINT ["python", "predict.py"]
CMD ["--help"]
