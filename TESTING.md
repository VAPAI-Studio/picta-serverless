# Testing MODNET Functionality

This guide helps you test the MODNET ONNX model download and inference functionality locally.

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Build the container
docker-compose build

# Run the services
docker-compose up -d

# Check logs
docker-compose logs -f worker-comfyui
```

### 2. Test MODNET Functionality

```bash
# Run the test script inside the container
docker-compose exec worker-comfyui python /test-modnet.py

# Or run it locally (if you have the dependencies)
python test-modnet.py
```

### 3. Manual Testing

You can also test manually:

```bash
# Check if the model was downloaded correctly
docker-compose exec worker-comfyui ls -lh /models/modnet/

# Test inference with a sample image
docker-compose exec worker-comfyui python -c "
from modnet_bg import get_person_mask_modnet_onnx
import cv2
import numpy as np

# Create a test image
img = np.random.rand(512, 512, 3).astype(np.float32)
mask = get_person_mask_modnet_onnx(img)
print(f'Mask shape: {mask.shape}, range: [{mask.min():.3f}, {mask.max():.3f}]')
"
```

## Environment Variables

The docker-compose file includes these key environment variables:

- `MODNET_ONNX_PATH`: Path where the ONNX model should be stored
- `MODEL_URL`: Direct URL to download the ONNX model from
- `ONNXRUNTIME_FORCE_CPU`: Set to "1" to force CPU-only execution
- `HF_TOKEN`: HuggingFace token for private models (optional)

## Troubleshooting

### Model Download Issues

If the model fails to download, check:

1. **Network connectivity**: Ensure the container can access the internet
2. **URL validity**: Verify the `MODEL_URL` points to a real ONNX file
3. **Authentication**: For private models, set `HF_TOKEN`

```bash
# Test URL accessibility
docker-compose exec worker-comfyui curl -I $MODEL_URL
```

### ONNX Runtime Issues

If you get ONNX Runtime errors:

1. **Check model file**: Verify it's a real ONNX file, not an LFS pointer
2. **File size**: Should be >1MB for a real model
3. **File type**: Should be binary, not HTML/text

```bash
# Check model file details
docker-compose exec worker-comfyui file /models/modnet/modnet.onnx
docker-compose exec worker-comfyui head -c 200 /models/modnet/modnet.onnx | od -An -tx1
```

### GPU Issues

If CUDA is not working:

1. **Check GPU availability**: Ensure nvidia-docker is installed
2. **Force CPU**: Set `ONNXRUNTIME_FORCE_CPU=1`
3. **Check drivers**: Verify NVIDIA drivers are compatible

```bash
# Check GPU availability
docker-compose exec worker-comfyui nvidia-smi

# Check ONNX Runtime providers
docker-compose exec worker-comfyui python -c "
import onnxruntime as ort
print('Available providers:', ort.get_available_providers())
"
```

## Expected Output

When everything works correctly, you should see:

```
[MODNET] expecting model at: /models/modnet/modnet.onnx
[MODNET] file info:
-rw-r--r-- 1 root root 89M /models/modnet/modnet.onnx
[MODNET] first bytes:
0000000 4f 4e 4e 58 00 00 00 00 00 00 00 00 00 00 00 00
[MODNET] file type:
/models/modnet/modnet.onnx: data
[MODNET] Attempting to load with providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
[MODNET] Successfully loaded with providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Clean Up

```bash
# Stop services
docker-compose down

# Remove volumes (this will delete downloaded models)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```
