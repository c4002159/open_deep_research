# Image used by code_sandbox.py's Docker backend (CODE_SANDBOX_IMAGE).
#   docker build -t agent-sandbox-python:latest -f docker/sandbox.Dockerfile .
# The container runs with no network, 256 MB RAM, 1 CPU, dropped capabilities.
FROM python:3.11-slim
RUN pip install --no-cache-dir pandas numpy scipy
RUN useradd -m sandbox
USER sandbox
WORKDIR /sandbox
