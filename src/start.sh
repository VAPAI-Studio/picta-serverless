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
# IMPORTANT: pass server args after `--` so they reach python main.py
comfy --workspace /comfyui launch -- --listen 127.0.0.1 --port 8188 --verbose "${COMFY_LOG_LEVEL}" >"$LOG_FILE" 2>&1 &
COMFY_PID=$!

# Wait for readiness (up to ~120s), or print crash logs
for i in {1..120}; do
  if curl -fsS http://127.0.0.1:8188/ >/dev/null; then
    echo "[READY] ComfyUI is up (pid=$COMFY_PID)"
    break
  fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "[FATAL] ComfyUI exited while starting. Last 120 log lines:"
    tail -n 120 "$LOG_FILE" || true
    exit 1
  fi
  sleep 1
done

echo "worker-comfyui: Starting RunPod Handler"
if [ "${SERVE_API_LOCALLY:-false}" = "true" ]; then
  exec python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
  exec python -u /handler.py
fi
