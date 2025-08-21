# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage (defaults provided by docker-bake.hcl)
ARG COMFYUI_VERSION=0.3.49
ARG CUDA_VERSION_FOR_COMFY=12.6
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126


# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
# Add root directory to Python path
ENV PYTHONPATH="/:${PYTHONPATH}"

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    wget \
    curl \
    file \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    build-essential \
    pkg-config \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && update-ca-certificates

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"
# Ensure root directory is in Python path for our modules
ENV PYTHONPATH="/:${PYTHONPATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
# RUN uv pip install comfy-cli pip "setuptools<75" wheel

# Install ComfyUI
# RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
#       /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
#     else \
#       /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
#     fi

# Upgrade PyTorch if needed (for newer CUDA versions)
# RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
#       uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
#     fi

# Change working directory to ComfyUI
# WORKDIR /comfyui

# Support for the network volume
# ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
COPY requirements.txt .

# Install ONNX Runtime GPU with fallback to CPU
RUN uv pip install "onnxruntime-gpu>=1.18,<1.19" || uv pip install "onnxruntime>=1.18,<1.19"

# Si usás BuildKit y necesitas SSH for deps privadas (git@...), podés activar la línea comentada de --mount=type=ssh
# RUN --mount=type=cache,target=/root/.cache/uv --mount=type=ssh \
RUN set -eux; \
    which uv; uv --version || true; \
    python -V; which python; which pip; \
    echo "----- First 200 lines of requirements.txt -----"; \
    sed -n '1,200p' requirements.txt || true; \
    echo "----------------------------------------------"; \
    # Verbose para ver exactamente en qué paquete/índice truena
    UV_VERBOSE=3 uv pip install -r requirements.txt

# Model directory
RUN mkdir -p /models/modnet

# Download MODNet ONNX model for serverless deployment
RUN set -e && \
    echo "Downloading MODNet ONNX model..." && \
    (wget -q -O /models/modnet/modnet.onnx \
        "https://huggingface.co/gradio/Modnet/resolve/main/modnet.onnx" || \
     wget -q -O /models/modnet/modnet.onnx \
        "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx" || \
     wget -q -O /models/modnet/modnet.onnx \
        "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/modnet.onnx" || \
     (echo "Error: Could not download MODNet model from any source" && exit 1)) && \
    echo "Verifying downloaded model..." && \
    [ -f /models/modnet/modnet.onnx ] && \
    [ $(wc -c < /models/modnet/modnet.onnx) -gt 1000000 ] && \
    echo "MODNet model downloaded successfully: $(wc -c < /models/modnet/modnet.onnx) bytes"

ENV MODNET_ONNX_PATH=/models/modnet/modnet.onnx
ENV ONNXRUNTIME_FORCE_CPU=0
# Fallback URL for runtime download if model is missing/corrupted
ENV MODEL_URL=https://huggingface.co/gradio/Modnet/resolve/main/modnet.onnx






# Add application code and scripts
ADD src/start.sh ./
COPY handler.py comfyui_handler.py recolor_handler.py modnet_bg.py ./
RUN chmod +x /start.sh

# Debug: List files to ensure they're copied correctly
RUN ls -la /*.py
RUN pwd && python -c "import sys; print('PYTHONPATH:', sys.path)"

# Add script to install custom nodes
# COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
# RUN chmod +x /usr/local/bin/comfy-node-install

# ENV COMFY_WORKSPACE=/comfyui
# RUN cd /comfyui && comfy-node-install comfyui-easy-use was-node-suite-comfyui


# deps típicas (additional ones not in requirements.txt)
# RUN uv pip install scipy einops


# Prevent pip from asking for confirmation during uninstall steps in custom nodes
# ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
# COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
# RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
# FROM base AS downloader

# ARG HUGGINGFACE_ACCESS_TOKEN
# # Set default model type if none is provided
# ARG MODEL_TYPE=flux1-dev

# # Change working directory to ComfyUI
# WORKDIR /comfyui

# # Create necessary directories upfront
# RUN mkdir -p models/checkpoints models/vae models/unet models/clip

# # Download checkpoints/vae/unet/clip models to include in image based on model type
# # RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
# #       wget -q -O models/unet/flux1-kontext-dev-fp8-e4m3fn.safetensors https://huggingface.co/6chan/flux1-kontext-dev-fp8/resolve/main/flux1-kontext-dev-fp8-e4m3fn.safetensors && \
# #       wget -q -O models/loras/aura_remove_1200x800_comfy.safetensors https://huggingface.co/yvesfogel/picta_auraremove_1200x800/resolve/main/aura_remove_1200x800_comfy.safetensors && \
# #       wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
# #       wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
# #       wget -q -O models/vae/ae.safetensors https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors; \
# #     fi


# Stage 3: Final image
# FROM base AS final

# Copy models from stage 2 to the final image
# COPY --from=downloader /comfyui/models /comfyui/models