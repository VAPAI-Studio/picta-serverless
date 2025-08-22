# ---------- Stage: base (build everything except *models*) ----------
    ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
    FROM ${BASE_IMAGE} AS base
    
    ARG COMFYUI_VERSION=0.3.51
    ARG CUDA_VERSION_FOR_COMFY=12.6
    ARG ENABLE_PYTORCH_UPGRADE=false
    ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
    
    ENV DEBIAN_FRONTEND=noninteractive
    ENV PIP_PREFER_BINARY=1
    ENV PYTHONUNBUFFERED=1
    ENV CMAKE_BUILD_PARALLEL_LEVEL=8
    ENV PYTHONPATH="/:${PYTHONPATH}"
    
    # Base deps
    RUN apt-get update && apt-get install -y \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        git wget curl file libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
        ffmpeg build-essential pkg-config ca-certificates \
     && update-ca-certificates \
     && ln -sf /usr/bin/python3.12 /usr/bin/python \
     && ln -sf /usr/bin/pip3 /usr/bin/pip \
     && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
    
    # uv + venv
    RUN wget -qO- https://astral.sh/uv/install.sh | sh \
     && ln -s /root/.local/bin/uv /usr/local/bin/uv \
     && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
     && uv venv /opt/venv
    ENV PATH="/opt/venv/bin:${PATH}"
    ENV PYTHONPATH="/:${PYTHONPATH}"
    
    # comfy-cli + ComfyUI
    RUN uv pip install comfy-cli pip "setuptools<75" wheel
    RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
          /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
        else \
          /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
        fi
    
    # Verify ComfyUI
    RUN if [ -f /comfyui/main.py ] || [ -f /comfyui/ComfyUI/main.py ]; then \
          echo "ComfyUI found"; \
        else \
          echo "ERROR: ComfyUI/main.py not found" && ls -la /comfyui && exit 1; \
        fi
    
    # Optional torch upgrade
    RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
          uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
        fi
    
    # Runtime deps
    WORKDIR /
    COPY requirements.txt .
    RUN set -eux; \
        which uv; uv --version || true; \
        python -V; which python; which pip; \
        sed -n '1,200p' requirements.txt || true; \
        UV_VERBOSE=3 uv pip install -r requirements.txt
    
    # ONNX Runtime + MODNet (small model OK to bake into image)
    RUN uv pip install "onnxruntime-gpu" || uv pip install "onnxruntime"
    RUN mkdir -p /models/modnet && set -e; \
        (wget -q -O /models/modnet/modnet.onnx \
          "https://huggingface.co/gradio/Modnet/resolve/main/modnet.onnx" || \
         wget -q -O /models/modnet/modnet.onnx \
          "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx" || \
         wget -q -O /models/modnet/modnet.onnx \
          "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/modnet.onnx" || \
         (echo "Error: Could not download MODNet model" && exit 1)) && \
        [ -f /models/modnet/modnet.onnx ] && \
        [ $(wc -c < /models/modnet/modnet.onnx) -gt 1000000 ]
    ENV MODNET_ONNX_PATH=/models/modnet/modnet.onnx
    ENV ONNXRUNTIME_FORCE_CPU=0
    ENV MODEL_URL=https://huggingface.co/gradio/Modnet/resolve/main/modnet.onnx
    ENV OMP_NUM_THREADS=1
    ENV CUDA_MODULE_LOADING=LAZY
    
    # App code
    ADD src/start.sh /start.sh
    COPY handler.py comfyui_handler.py recolor_handler.py modnet_bg.py ./
    RUN chmod +x /start.sh \
     && ls -la /*.py \
     && python -c "import sys; print('PYTHONPATH:', sys.path)"
    
    # Custom nodes (kept in image; models pulled from network volume)
    COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
    RUN chmod +x /usr/local/bin/comfy-node-install
    ENV COMFY_WORKSPACE=/comfyui
    RUN cd /comfyui && comfy-node-install comfyui-easy-use was-node-suite-comfyui || echo "[WARN] Some custom nodes failed to install"
    
    # Manager helper
    COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
    RUN sed -i 's/\r$//' /start.sh /usr/local/bin/comfy-manager-set-mode \
     && chmod +x /start.sh /usr/local/bin/comfy-manager-set-mode
    
    # ---------- Stage: final (add only runtime configs relying on network volume) ----------
    FROM base AS final
    
    # Point ComfyUI to the mounted network volume paths
    ADD src/extra_model_paths.yaml /comfyui/extra_model_paths.yaml
    
    # Optional: bootstrap to ensure dirs / first-time seed if you want that behavior
    ADD src/bootstrap_models.sh /usr/local/bin/bootstrap-models
    RUN chmod +x /usr/local/bin/bootstrap-models
    
    # Final CMD (single source of truth)
    CMD ["/start.sh"]
    