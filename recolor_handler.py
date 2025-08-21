import json
import base64
import numpy as np
import cv2
from skimage import color
from skimage.color import deltaE_ciede2000
from transparent_background import Remover
from PIL import Image

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

def hex_to_rgb01(hx: str) -> np.ndarray:
    hx = hx.strip()
    if hx.startswith("#"):
        hx = hx[1:]
    if len(hx) != 6:
        raise ValueError("HEX must be 6 chars, e.g. #f5f9f5")
    return np.array([int(hx[i:i+2], 16) for i in (0, 2, 4)]) / 255.0


def trimmed_mean(arr: np.ndarray, p: float) -> float:
    if p <= 0:
        return np.median(arr)
    n = arr.size
    if n == 0:
        return 0.0
    k = int(n * p)
    if 2 * k >= n:
        return float(arr.mean())
    s = np.sort(arr)
    return float(s[k:n - k].mean())


def robust_stat(arr: np.ndarray, p: float) -> float:
    return trimmed_mean(arr, p) if p > 0 else float(np.median(arr))


def improve_mask_consistency(mask: np.ndarray, min_area: int = 1000) -> np.ndarray:
    mask_bin = (mask > 0.5).astype(np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask_bin[labels == i] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    
    mask_smooth = cv2.GaussianBlur(mask_bin.astype(np.float32), (9, 9), 2)
    
    return np.clip(mask_smooth, 0, 1)


# one global instance (loads weights once)
_tb_remover = Remover()  # uses InSPyReNet; returns RGBA/masks

def get_person_mask_tb(img_rgb: np.ndarray) -> np.ndarray:
    """
    Drop-in replacement for get_person_mask_rembg.
    img_rgb: float32, [0,1], HxWx3
    returns: float32 mask in [0,1], HxW
    """
    img_uint8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    # 'rgba' -> returns RGBA image; alpha is the matte
    out = _tb_remover.process(pil_img, type='rgba')
    out_np = np.asarray(out)
    if out_np.shape[-1] == 4:
        mask = out_np[..., 3] / 255.0
    else:
        # fallback: if only RGB is returned, treat non-black as FG
        mask = (out_np.sum(axis=-1) > 0).astype(np.float32)
    return improve_mask_consistency(mask)


def recolor_background_consistent(img_rgb: np.ndarray,
                                 mask_person: np.ndarray,
                                 target_hex: str = "#f5f9f5",
                                 delta_e_tol: float = 3.0,
                                 trim_perc: float = 0.1,
                                 alpha_chroma: float = 1.0,
                                 use_fixed_reference: bool = True):
    mask_bg = 1.0 - mask_person

    lab = color.rgb2lab(img_rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    bg_idx = mask_bg > 0.5
    if not np.any(bg_idx):
        raise ValueError("No background pixels found. Check the mask.")

    L_bg, a_bg, b_bg = L[bg_idx], a[bg_idx], b[bg_idx]

    t_rgb = hex_to_rgb01(target_hex)
    t_lab = color.rgb2lab(t_rgb[np.newaxis, np.newaxis, :])[0, 0]
    L_t, a_t, b_t = t_lab

    if use_fixed_reference:
        L_ref = 85.0
        a_ref = 0.0
        b_ref = 0.0
        
        scale_L = L_t / (L_ref + 1e-6)
        da = a_t - a_ref
        db = b_t - b_ref
    else:
        L_med = robust_stat(L_bg, trim_perc)
        a_med = robust_stat(a_bg, trim_perc)
        b_med = robust_stat(b_bg, trim_perc)
        
        scale_L = L_t / (L_med + 1e-6)
        da = a_t - a_med
        db = b_t - b_med

    L_new = L.copy()
    L_new[bg_idx] = np.clip(L_bg * scale_L, 0, 100)

    a_new = a.copy()
    b_new = b.copy()
    a_new[bg_idx] = np.clip(a_bg + alpha_chroma * da, -128, 127)
    b_new[bg_idx] = np.clip(b_bg + alpha_chroma * db, -128, 127)
    
    L_target = L_t
    a_target = a_t
    b_target = b_t
    
    L_range = 3.0
    a_range = 1.5
    b_range = 1.5
    
    L_new[bg_idx] = np.clip(L_new[bg_idx], L_target - L_range, L_target + L_range)
    a_new[bg_idx] = np.clip(a_new[bg_idx], a_target - a_range, a_target + a_range)
    b_new[bg_idx] = np.clip(b_new[bg_idx], b_target - b_range, b_target + b_range)

    lab_new = np.stack([L_new, a_new, b_new], axis=-1)
    rgb_new = np.clip(color.lab2rgb(lab_new), 0, 1)

    out = mask_person[..., None] * img_rgb + (1 - mask_person)[..., None] * rgb_new

    target_map = np.zeros_like(lab_new)
    target_map[..., 0] = L_t
    target_map[..., 1] = a_t
    target_map[..., 2] = b_t
    dE_map = deltaE_ciede2000(lab_new, target_map)

    dE_bg = dE_map[bg_idx]
    
    L_result = robust_stat(L_new[bg_idx], trim_perc)
    a_result = robust_stat(a_new[bg_idx], trim_perc)
    b_result = robust_stat(b_new[bg_idx], trim_perc)
    
    L_result_std = float(np.std(L_new[bg_idx]))
    a_result_std = float(np.std(a_new[bg_idx]))
    b_result_std = float(np.std(b_new[bg_idx]))
    
    L_result_p25 = float(np.percentile(L_new[bg_idx], 25))
    L_result_p75 = float(np.percentile(L_new[bg_idx], 75))
    a_result_p25 = float(np.percentile(a_new[bg_idx], 25))
    a_result_p75 = float(np.percentile(a_new[bg_idx], 75))
    b_result_p25 = float(np.percentile(b_new[bg_idx], 25))
    b_result_p75 = float(np.percentile(b_new[bg_idx], 75))
    
    L_med_orig = robust_stat(L_bg, trim_perc)
    a_med_orig = robust_stat(a_bg, trim_perc)
    b_med_orig = robust_stat(b_bg, trim_perc)
    
    metrics = {
        "L_med_orig": float(L_med_orig),
        "a_med_orig": float(a_med_orig),
        "b_med_orig": float(b_med_orig),
        "L_ref": float(L_ref if use_fixed_reference else L_med_orig),
        "a_ref": float(a_ref if use_fixed_reference else a_med_orig),
        "b_ref": float(b_ref if use_fixed_reference else b_med_orig),
        "L_result": float(L_result),
        "a_result": float(a_result),
        "b_result": float(b_result),
        "L_result_std": L_result_std,
        "a_result_std": a_result_std,
        "b_result_std": b_result_std,
        "L_result_p25": L_result_p25,
        "L_result_p75": L_result_p75,
        "a_result_p25": a_result_p25,
        "a_result_p75": a_result_p75,
        "b_result_p25": b_result_p25,
        "b_result_p75": b_result_p75,
        "scale_L": float(scale_L),
        "da": float(da),
        "db": float(db),
        "dE_mean": float(dE_bg.mean()),
        "dE_p95": float(np.percentile(dE_bg, 95)),
        "warn_delta_e": float(dE_bg.mean()) > delta_e_tol,
        "use_fixed_reference": use_fixed_reference
    }

    return out, lab_new, metrics


def process_image_recolor(img_rgb: np.ndarray,
                         target_hex: str = "#f5f9f5",
                         delta_e_tol: float = 3.0,
                         trim_perc: float = 0.1,
                         alpha_chroma: float = 1.0,
                         use_consistent_bg: bool = True):
    
    mask_person = get_person_mask_tb(img_rgb)

    if use_consistent_bg:
        recolored, lab_new, metrics = recolor_background_consistent(
            img_rgb, mask_person,
            target_hex=target_hex,
            delta_e_tol=delta_e_tol,
            trim_perc=trim_perc,
            alpha_chroma=alpha_chroma,
            use_fixed_reference=True
        )
    else:
        # Use the original method if needed
        mask_bg = 1.0 - mask_person
        lab = color.rgb2lab(img_rgb)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        bg_idx = mask_bg > 0.5
        if not np.any(bg_idx):
            raise ValueError("No background pixels found. Check the mask.")
        
        L_bg, a_bg, b_bg = L[bg_idx], a[bg_idx], b[bg_idx]
        L_med = robust_stat(L_bg, trim_perc)
        a_med = robust_stat(a_bg, trim_perc)
        b_med = robust_stat(b_bg, trim_perc)
        
        t_rgb = hex_to_rgb01(target_hex)
        t_lab = color.rgb2lab(t_rgb[np.newaxis, np.newaxis, :])[0, 0]
        L_t, a_t, b_t = t_lab
        
        scale_L = L_t / (L_med + 1e-6)
        L_new = L.copy()
        L_new[bg_idx] = np.clip(L_bg * scale_L, 0, 100)
        
        da, db = a_t - a_med, b_t - b_med
        a_new = a.copy()
        b_new = b.copy()
        a_new[bg_idx] = np.clip(a_bg + alpha_chroma * da, -128, 127)
        b_new[bg_idx] = np.clip(b_bg + alpha_chroma * db, -128, 127)
        
        # Apply color consistency fix
        L_target = L_t
        a_target = a_t
        b_target = b_t
        
        L_range = 3.0
        a_range = 1.5
        b_range = 1.5
        
        L_new[bg_idx] = np.clip(L_new[bg_idx], L_target - L_range, L_target + L_range)
        a_new[bg_idx] = np.clip(a_new[bg_idx], a_target - a_range, a_target + a_range)
        b_new[bg_idx] = np.clip(b_new[bg_idx], b_target - b_range, b_target + b_range)
        
        lab_new = np.stack([L_new, a_new, b_new], axis=-1)
        rgb_new = np.clip(color.lab2rgb(lab_new), 0, 1)
        
        recolored = mask_person[..., None] * img_rgb + (1 - mask_person)[..., None] * rgb_new
        
        # Calculate metrics
        target_map = np.zeros_like(lab_new)
        target_map[..., 0] = L_t
        target_map[..., 1] = a_t
        target_map[..., 2] = b_t
        dE_map = deltaE_ciede2000(lab_new, target_map)
        dE_bg = dE_map[bg_idx]
        
        L_result = robust_stat(L_new[bg_idx], trim_perc)
        a_result = robust_stat(a_new[bg_idx], trim_perc)
        b_result = robust_stat(b_new[bg_idx], trim_perc)
        
        metrics = {
            "L_med": float(L_med),
            "a_med": float(a_med),
            "b_med": float(b_med),
            "L_result": float(L_result),
            "a_result": float(a_result),
            "b_result": float(b_result),
            "scale_L": float(scale_L),
            "da": float(da),
            "db": float(db),
            "dE_mean": float(dE_bg.mean()),
            "dE_p95": float(np.percentile(dE_bg, 95)),
            "warn_delta_e": float(dE_bg.mean()) > delta_e_tol
        }

    return recolored, metrics


def validate_recolor_input(job_input):
    """Validates input for background recoloring"""
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Check for required image parameter
    image_data = job_input.get("image")
    if image_data is None:
        return None, "Missing 'image' parameter"

    # Optional parameters with defaults
    target_hex = job_input.get("target_hex", "#f5f9f5")
    delta_e_tol = job_input.get("delta_e_tol", 3.0)
    trim_perc = job_input.get("trim_perc", 0.1)
    alpha_chroma = job_input.get("alpha_chroma", 1.0)
    use_consistent_bg = job_input.get("use_consistent_bg", True)

    return {
        "image": image_data,
        "target_hex": target_hex,
        "delta_e_tol": delta_e_tol,
        "trim_perc": trim_perc,
        "alpha_chroma": alpha_chroma,
        "use_consistent_bg": use_consistent_bg
    }, None


def handle_recolor_background(job_input, job_id):
    """Handler for background recoloring functionality"""
    try:
        validated_data, error_message = validate_recolor_input(job_input)
        if error_message:
            return {"error": error_message}

        # Extract image data and decode base64
        image_data_uri = validated_data["image"]
        
        # Strip Data URI prefix if present
        if "," in image_data_uri:
            base64_data = image_data_uri.split(",", 1)[1]
        else:
            base64_data = image_data_uri
        
        # Clean and pad the base64 string
        base64_data = base64_data.strip()
        
        # Validate base64 string length
        if len(base64_data) == 0:
            return {"error": "Empty base64 string provided"}
        
        # Add padding if necessary
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += '=' * (4 - missing_padding)
        
        # Decode the image
        image_bytes = base64.b64decode(base64_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Could not decode image data"}
        
        # Convert BGR to RGB and normalize to [0,1]
        img_rgb = img[..., ::-1].astype(np.float32) / 255.0

        # Process the image
        recolored, metrics = process_image_recolor(
            img_rgb,
            target_hex=validated_data["target_hex"],
            delta_e_tol=validated_data["delta_e_tol"],
            trim_perc=validated_data["trim_perc"],
            alpha_chroma=validated_data["alpha_chroma"],
            use_consistent_bg=validated_data["use_consistent_bg"]
        )

        # Convert back to uint8 and BGR for OpenCV
        result_img = (np.clip(recolored, 0, 1) * 255).astype(np.uint8)
        result_img_bgr = result_img[..., ::-1]

        # Encode as PNG
        success, encoded_img = cv2.imencode('.png', result_img_bgr)
        if not success:
            return {"error": "Failed to encode processed image"}

        # Convert to base64
        result_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

        return {
            "image": result_base64,
            "metrics": metrics,
            "status": "success"
        }

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}