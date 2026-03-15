.PHONY: sync sync-gpu verify-gpu clean

# Standard CPU sync (no GPU)
sync:
	uv sync

# GPU sync: uv sync then force-reinstall onnxruntime-gpu to ensure the GPU
# cpython binding overwrites the CPU one that magika pulls in via chonkie[code]
sync-gpu:
	uv sync
	uv pip install --force-reinstall "onnxruntime-gpu[cuda,cudnn]>=1.24.0"

# Verify CUDA/TensorRT providers are available
verify-gpu:
	uv run python -c "\
import onnxruntime as ort; \
p = ort.get_available_providers(); \
print('Version:', ort.__version__); \
print('Providers:', p); \
print('CUDA:', 'CUDAExecutionProvider' in p); \
print('TensorRT:', 'TensorrtExecutionProvider' in p)"

clean:
	rm -rf .venv dist build __pycache__ *.pyc
