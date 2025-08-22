#!/usr/bin/env bash
set -euo pipefail

# Required env vars:
: "${VOL:?Set VOL to your Network Volume ID (e.g., ti0ck2hqze)}"
: "${REG:?Set REG to your Runpod region (e.g., eur-is-1)}"
: "${EP:?Set EP to your Runpod S3 endpoint URL (e.g., https://s3api-eur-is-1.runpod.io)}"

# If you keep local copies, place them under ./models/...
LOCAL_ROOT="${LOCAL_ROOT:-./models}"
S3_ROOT="s3://${VOL}/comfyui/models"

# Prefer curl, fallback to wget
have_curl=0
if command -v curl >/dev/null 2>&1; then have_curl=1; fi

need_tool() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: '$1' is required but not installed." >&2
    exit 1
  }
}

need_tool aws
if [ $have_curl -eq 0 ]; then need_tool wget; fi

# Create a temp workspace for downloads
TMPDIR="$(mktemp -d -t rpseed-XXXXXXXX)"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

# Map of TARGET_KEY => SOURCE (local path preferred; otherwise URL will be used)
# TARGET_KEY is the path relative to S3_ROOT
declare -A TARGETS

TARGETS["unet/flux1-kontext-dev-fp8-e4m3fn.safetensors"]="$LOCAL_ROOT/unet/flux1-kontext-dev-fp8-e4m3fn.safetensors|https://huggingface.co/6chan/flux1-kontext-dev-fp8/resolve/main/flux1-kontext-dev-fp8-e4m3fn.safetensors"
TARGETS["loras/aura_remove_1200x800_comfy.safetensors"]="$LOCAL_ROOT/loras/aura_remove_1200x800_comfy.safetensors|https://huggingface.co/yvesfogel/picta_auraremove_1200x800/resolve/main/aura_remove_1200x800_comfy.safetensors"
TARGETS["clip/clip_l.safetensors"]="$LOCAL_ROOT/clip/clip_l.safetensors|https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
TARGETS["clip/t5xxl_fp8_e4m3fn.safetensors"]="$LOCAL_ROOT/clip/t5xxl_fp8_e4m3fn.safetensors|https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
TARGETS["vae/ae.safetensors"]="$LOCAL_ROOT/vae/ae.safetensors|https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors"
TARGETS["diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors"]="$LOCAL_ROOT/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors|https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors"
TARGETS["loras/Qwen-Image-Lightning-4steps-V1.0.safetensors"]="$LOCAL_ROOT/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors|https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors"
TARGETS["text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"]="$LOCAL_ROOT/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors|https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
TARGETS["vae/qwen_image_vae.safetensors"]="$LOCAL_ROOT/vae/qwen_image_vae.safetensors|https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"

# Helper: upload a file to S3 destination
upload() {
  local src="$1"
  local key="$2"      # e.g., unet/foo.safetensors
  local dst="${S3_ROOT}/${key}"

  echo "→ Uploading: ${src}  →  ${dst}"
  aws s3 cp "$src" "$dst" --region "$REG" --endpoint-url "$EP" --no-progress
}

# If you want to pre-create "folders" (not strictly needed), you can drop empty markers:
# printf '' | aws s3 cp - "${S3_ROOT}/.keep" --region "$REG" --endpoint-url "$EP" || true

uploaded=()
skipped=()
failed=()

for key in "${!TARGETS[@]}"; do
  IFS='|' read -r local_path url <<< "${TARGETS[$key]}"

  # Decide source path: local if exists, else download
  src=""
  if [ -s "$local_path" ]; then
    echo "[ok] Found local file: $local_path"
    src="$local_path"
  else
    echo "[..] Local missing, downloading: $url"
    fname="$(basename "$key")"
    tmp_path="${TMPDIR}/${fname}"

    if [ $have_curl -eq 1 ]; then
      # -L follow redirects, -f fail on HTTP errors, -C - resume, --retry for robustness
      if curl -L -f --retry 3 --retry-delay 2 -o "$tmp_path" "$url"; then
        src="$tmp_path"
      else
        echo "[ERR] Download failed: $url"
        failed+=("$key")
        continue
      fi
    else
      if wget -q -O "$tmp_path" "$url"; then
        src="$tmp_path"
      else
        echo "[ERR] Download failed: $url"
        failed+=("$key")
        continue
      fi
    fi
  fi

  # Sanity check: file non-empty
  if [ ! -s "$src" ]; then
    echo "[ERR] Source empty: $src"
    failed+=("$key")
    continue
  fi

  # Upload
  if upload "$src" "$key"; then
    uploaded+=("$key")
  else
    echo "[ERR] Upload failed: $key"
    failed+=("$key")
  fi
done

echo
echo "==== Summary ===="
echo "Uploaded: ${#uploaded[@]}"
for k in "${uploaded[@]}"; do echo "  - $k"; done

if [ ${#failed[@]} -gt 0 ]; then
  echo "Failed: ${#failed[@]}"
  for k in "${failed[@]}"; do echo "  - $k"; done
  exit 2
fi

echo "All done."
