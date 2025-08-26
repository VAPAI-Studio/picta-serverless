#!/usr/bin/env bash
set -Eeuo pipefail

# --- Optional: use tcmalloc only if present (use full path) ---
if ldconfig -p | grep -q 'libtcmalloc.so'; then
  export LD_PRELOAD="$(ldconfig -p | awk '/libtcmalloc\.so/{print $NF; exit}')"
fi

# Manager offline mode (best-effort)
if command -v comfy-manager-set-mode >/dev/null 2>&1; then
  comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2
fi

export PYTHONPATH="${PYTHONPATH:-}:/"
: "${COMFY_LOG_LEVEL:=DEBUG}"
LOG_FILE=/tmp/comfy.log

echo "worker-comfyui: Starting ComfyUI (workspace=/comfyui)"
echo "worker-comfyui: Log level: ${COMFY_LOG_LEVEL}"
echo "worker-comfyui: Command: comfy --workspace /comfyui launch -- --listen 127.0.0.1 --port 8188 --verbose ${COMFY_LOG_LEVEL}"
# IMPORTANT: pass server args after `--` so they reach python main.py
comfy --workspace /comfyui launch -- --listen 127.0.0.1 --port 8188 --verbose "${COMFY_LOG_LEVEL}" >"$LOG_FILE" 2>&1 &
COMFY_PID=$!
echo "worker-comfyui: ComfyUI started with PID: $COMFY_PID"

# Wait for readiness (up to ~120s), or print crash logs
for i in {1..120}; do
  if curl -fsS http://127.0.0.1:8188/ >/dev/null; then
    echo "[READY] ComfyUI is up (pid=$COMFY_PID)"
    break
  fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "[FATAL] ComfyUI exited while starting (attempt $i/120). PID $COMFY_PID is no longer running."
    echo "[FATAL] Checking log file: $LOG_FILE"
    if [ -f "$LOG_FILE" ]; then
      echo "[FATAL] Log file exists. Last 120 log lines:"
      tail -n 120 "$LOG_FILE" || echo "[ERROR] Could not read log file"
    else
      echo "[FATAL] Log file does not exist at $LOG_FILE"
      echo "[FATAL] Checking if comfy command exists:"
      which comfy || echo "comfy command not found"
      echo "[FATAL] Trying to run comfy directly for debugging:"
      comfy --version || echo "comfy --version failed"
    fi
    exit 1
  fi
  if [ $((i % 10)) -eq 0 ]; then
    echo "worker-comfyui: Still waiting for ComfyUI to start (attempt $i/120)..."
  fi
  sleep 1
done

echo "worker-comfyui: Starting RunPod Handler"
if [ "${SERVE_API_LOCALLY:-false}" = "true" ]; then
  exec python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
  exec python -u /handler.py
fi
