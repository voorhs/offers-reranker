FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir sentence-transformers==5.1.2

# Copy prediction script
COPY predict/predict.py .

# Copy model (run docker build from repo root with context)
COPY models/rubert-tiny-reranker /model

ENTRYPOINT ["python", "predict.py"]
CMD ["--help"]

