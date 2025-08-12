# ---- arriba de todo
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
FROM ${BASE_IMAGE} AS base

# Defaults seguros (cámbialos si querés)
ARG COMFYUI_VERSION=0.2.1
ARG CUDA_VERSION_FOR_COMFY=12.6
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

# ... (apt, uv, etc. igual)

RUN uv pip install "comfy-cli==0.9.1" pip setuptools wheel

# Debug + skip gpu check (en build no hay GPU)
RUN set -eux; \
    echo "COMFYUI_VERSION=${COMFYUI_VERSION}  CUDA=${CUDA_VERSION_FOR_COMFY}"; \
    /usr/bin/yes | comfy --workspace /comfyui install \
      --version "${COMFYUI_VERSION}" \
      --nvidia \
      --cuda-version "${CUDA_VERSION_FOR_COMFY}" \
      --skip-gpu-check || { echo 'Comfy install failed'; exit 1; }

# (opcional) Fuerza Torch con cu126 si después querés
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi
