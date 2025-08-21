import os
import sys
import cv2
import numpy as np

# Import from recolor_handler since that's where improve_mask_consistency is defined
try:
    from recolor_handler import improve_mask_consistency
except Exception:
    # fallback: local stub if you want to run this module standalone
    def improve_mask_consistency(mask: np.ndarray, min_area: int = 1000) -> np.ndarray:
        mask_bin = (mask > 0.5).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                mask_bin[labels == i] = 0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k)
        mask_smooth = cv2.GaussianBlur(mask_bin.astype(np.float32), (9, 9), 2)
        return np.clip(mask_smooth, 0, 1)

# --------------------------
# ONNXRuntime lazy loader
# --------------------------
_ort = None
_sess = None
_in_name = None
_out_name = None
_in_size = 512  # common export uses 512; adjust if your export differs

def _load_ort():
    global _ort
    if _ort is not None:
        return _ort
    import onnxruntime as ort
    _ort = ort
    return _ort

def load_modnet_session(modnet_onnx_path: str = None):
    """
    Create a global ONNXRuntime session with CUDA if available; otherwise CPU.
    """
    global _sess, _in_name, _out_name
    if _sess is not None:
        return _sess

    ort = _load_ort()
    modnet_onnx_path = modnet_onnx_path or os.environ.get("MODNET_ONNX_PATH", "/models/modnet/modnet.onnx")

    def _assert_model_looks_binary(path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MODNet ONNX missing at {path}")
        size = os.path.getsize(path)
        if size < 1_000_000:
            raise ValueError(f"ONNX too small ({size} bytes) — likely LFS pointer/HTML.")
        with open(path, "rb") as f:
            head = f.read(200)
        hlow = head.lower()
        if b"git-lfs" in hlow or b"<!doctype html" in hlow or b"<html" in hlow:
            raise ValueError("ONNX looks like LFS pointer/HTML page, not a model.")

    _assert_model_looks_binary(modnet_onnx_path)

    force_cpu = os.environ.get("ONNXRUNTIME_FORCE_CPU", "0") == "1"

    providers = []
    if not force_cpu:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    try:
        _sess = ort.InferenceSession(modnet_onnx_path, providers=providers)
    except Exception:
        # hard fallback to CPU if CUDA wheel/provider mismatches
        _sess = ort.InferenceSession(modnet_onnx_path, providers=["CPUExecutionProvider"])

    # Determine IO names dynamically
    _in_name = _sess.get_inputs()[0].name
    _out_name = _sess.get_outputs()[0].name

    # Try to infer network input size if the model is fixed-size
    try:
        shape = _sess.get_inputs()[0].shape  # e.g., [1,3,512,512]
        if isinstance(shape, (list, tuple)) and len(shape) == 4 and shape[2] and shape[3]:
            # shape can be None for dynamic; guard it
            global _in_size
            _in_size = int(shape[2])  # assume square
    except Exception:
        pass

    return _sess

def _letterbox_rgb01(img: np.ndarray, size: int):
    """
    Resize with aspect ratio, pad to square 'size' using edge padding.
    img: float32 [0,1], HxWx3
    returns: padded_img [size,size,3], (H,W), (newH,newW)
    """
    H, W = img.shape[:2]
    scale = size / float(min(H, W))
    newH, newW = int(round(H * scale)), int(round(W * scale))
    resized = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_AREA)
    padH, padW = size - newH, size - newW
    # pad bottom/right; edge pad to avoid seams
    padded = np.pad(resized, ((0, padH), (0, padW), (0, 0)), mode="edge")
    return padded, (H, W), (newH, newW)

def _to_nchw(img_rgb01: np.ndarray):
    """
    [H,W,3] -> [1,3,H,W] float32
    """
    return img_rgb01.transpose(2, 0, 1)[None].astype(np.float32)

def get_person_mask_modnet_onnx(img_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a portrait matte using MODNet ONNX.
    Input:  img_rgb float32 [0,1], HxWx3
    Output: mask float32 [0,1], HxW
    """
    sess = load_modnet_session()
    size = _in_size

    img_rgb = np.clip(img_rgb, 0, 1).astype(np.float32)
    padded, (H, W), (newH, newW) = _letterbox_rgb01(img_rgb, size)
    inp = _to_nchw(padded)

    # Optionally normalize if your export expects specific mean/std.
    # Many MODNet exports work fine in [0,1]. If needed:
    # inp = (inp - np.array([0.5,0.5,0.5])[None,:,None,None]) / np.array([0.5,0.5,0.5])[None,:,None,None]

    outputs = sess.run([_out_name], {_in_name: inp})
    matte = outputs[0]  # expect [1,1,H,W] or [1,H,W]
    if matte.ndim == 4:
        matte = matte[0, 0]
    elif matte.ndim == 3:
        matte = matte[0]
    else:
        raise RuntimeError(f"Unexpected MODNet output shape: {outputs[0].shape}")

    # crop away padding, resize back to original
    matte = matte[:newH, :newW]
    matte = cv2.resize(matte, (W, H), interpolation=cv2.INTER_LINEAR)

    # Optional gentle feather to suppress halos
    matte = cv2.GaussianBlur(matte, (0, 0), sigmaX=1.0, sigmaY=1.0)

    # Your existing post-processing to remove tiny blobs and smooth edges
    matte = improve_mask_consistency(np.clip(matte, 0, 1))

    return np.clip(matte, 0, 1)

# -------------
# CLI smoke test
# -------------
if __name__ == "__main__":
    import argparse
    from PIL import Image

    p = argparse.ArgumentParser()
    p.add_argument("image_path")
    p.add_argument("--out", default="mask.png")
    args = p.parse_args()

    img = Image.open(args.image_path).convert("RGB")
    img_rgb = np.asarray(img).astype(np.float32) / 255.0

    m = get_person_mask_modnet_onnx(img_rgb)
    m8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(args.out, m8)
    print(f"Saved matte to {args.out}")