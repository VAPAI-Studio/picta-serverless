#!/usr/bin/env bash
set -e

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
if command -v comfy-manager-set-mode >/dev/null 2>&1; then
    comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2
else
    echo "worker-comfyui - comfy-manager-set-mode not found in PATH" >&2
fi

echo "worker-comfyui: Starting ComfyUI"

# Debug: Validate ComfyUI installation
echo "[DEBUG] Checking ComfyUI installation..."
ls -la /comfyui/ || true
if [ -d /comfyui/ComfyUI ]; then
    ls -la /comfyui/ComfyUI/ | head || true
else
    echo "[INFO] ComfyUI installed directly in /comfyui/"
fi
command -v comfy || echo "[WARN] comfy-cli not in PATH"
command -v comfy-manager-set-mode || echo "[WARN] comfy-manager-set-mode not in PATH"

# Add current directory to Python path for module imports
export PYTHONPATH="${PYTHONPATH}:/"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

# ------------------------------------------------------
# MODNET: sanity checks and self-healing model download
# ------------------------------------------------------
echo "[MODNET] expecting model at: ${MODNET_ONNX_PATH:-/models/modnet/modnet.onnx}"
MODEL_PATH="${MODNET_ONNX_PATH:-/models/modnet/modnet.onnx}"

if [ -f "$MODEL_PATH" ]; then
    echo "[MODNET] file info:"
    ls -lh "$MODEL_PATH" || true
    echo "[MODNET] first bytes:"
    head -c 128 "$MODEL_PATH" | od -An -tx1 | head -n 2 || true
    echo "[MODNET] file type:"
    file "$MODEL_PATH" || true
else
    echo "[MODNET] model not found at $MODEL_PATH"
fi

needs_download=0
if [ ! -f "$MODEL_PATH" ]; then
    needs_download=1
else
    SIZE=$(wc -c < "$MODEL_PATH" || echo 0)
    if [ "$SIZE" -lt 1000000 ]; then
        needs_download=1
    elif head -c 200 "$MODEL_PATH" | grep -qi "git-lfs"; then
        needs_download=1
    elif head -c 200 "$MODEL_PATH" | grep -qi "<!DOCTYPE html>"; then
        needs_download=1
    fi
fi

if [ "$needs_download" -eq 1 ]; then
    echo "[MODNET] downloading model..."
    mkdir -p "$(dirname "$MODEL_PATH")"
    if [ -z "$MODEL_URL" ]; then
        echo "[MODNET] ERROR: MODEL_URL not set and local model invalid/missing." >&2
        exit 1
    fi
    if [ -n "$HF_TOKEN" ]; then
        curl -fL -H "Authorization: Bearer $HF_TOKEN" "$MODEL_URL" -o "$MODEL_PATH"
    else
        curl -fL "$MODEL_URL" -o "$MODEL_PATH"
    fi
    echo "[MODNET] download done:"
    ls -lh "$MODEL_PATH" || true
fi

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    echo "worker-comfyui: Starting ComfyUI via comfy-cli..."
    comfy --workspace /comfyui launch --listen 0.0.0.0 --port 8188 --verbose "${COMFY_LOG_LEVEL}" &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    echo "worker-comfyui: Starting ComfyUI via comfy-cli..."
    comfy --workspace /comfyui launch --listen 0.0.0.0 --port 8188 --verbose "${COMFY_LOG_LEVEL}" &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi