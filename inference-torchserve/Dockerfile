# TorchServe Docker image for offers reranker
FROM pytorch/torchserve:0.12.0-cpu

# Switch to root for installing dependencies
USER root

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment - install to shared location
ENV UV_PYTHON_INSTALL_DIR=/opt/python
ENV PATH="/opt/venv/bin:$PATH"

# Install Python 3.12 and create venv
RUN uv python install 3.12 && \
    uv venv /opt/venv --python 3.12 && \
    chmod -R a+rX /opt/python

# Copy dependency files
COPY pyproject.toml pyproject.toml

# Install pip and Python dependencies
RUN uv pip install --python /opt/venv/bin/python --no-cache pip -r pyproject.toml

# Make venv accessible to model-server user
RUN chown -R model-server:model-server /opt/venv

# Create model store directory
RUN mkdir -p /home/model-server/model-store && \
    chown -R model-server:model-server /home/model-server/model-store

# Copy configuration
COPY config.properties /home/model-server/config.properties

# Copy the pre-built model archive (built during docker build or mounted)
# The .mar file should be created by running scripts/package_model.sh first
COPY model-store/reranker.mar /home/model-server/model-store/reranker.mar

# Switch back to model-server user
USER model-server

# Expose ports
# 8080: Inference API
# 8081: Management API
# 8082: Metrics API
EXPOSE 8080 8081 8082

# Start TorchServe with the configuration
CMD ["torchserve", \
     "--start", \
     "--foreground", \
     "--model-store", "/home/model-server/model-store", \
     "--ts-config", "/home/model-server/config.properties", \
     "--disable-token-auth"]
