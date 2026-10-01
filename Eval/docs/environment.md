# Evaluation environment

Use Linux x86_64 and Python 3.10 (evaluated: 3.10.9). The evaluated CUDA stack is PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, CUDA runtime 12.4, cuDNN 9.1, and flash-attn 2.7.4.post1. PyTorch's bundled CUDA runtime still requires a compatible NVIDIA driver. The flash-attn wheel must match Python, PyTorch and its C++ ABI; the evaluated PyTorch wheel uses `cxx11abiFALSE`. Building flash-attn from source also requires a compatible CUDA development toolkit and compiler.

Follow the installation sequence in [README](../README.md): install PyTorch first, then build prerequisites, `requirements.txt` with `--no-build-isolation`, and finally `requirements-opencv.txt` with `--no-deps`. The latter flag applies only to the OpenCV file. Do not use `--no-deps` for the main requirements.

All explicitly pinned runtime packages match the evaluated environment, including Transformers 4.57.6, qwen-vl-utils 0.0.14, NumPy 1.26.4, pandas 2.3.2, PyAV 14.2.0, Decord 0.6.0, Pillow 12.0.0 and OpenAI 2.26.0. ImageIO 2.37.2 is declared because the retained video utility supports GIF input. These are runtime pins, not a complete transitive environment lock.

## Known metadata inconsistencies

The evaluated environment is operational but does not pass `pip check` without diagnostics:

1. **OpenCV 4.13.0.92 declares NumPy >=2 for Python >=3.9**, while evaluation used NumPy 1.26.4. Install the OpenCV wheel separately with `--no-deps` to reproduce these exact versions. A comparison against OpenCV 4.11.0.86 found different decoded pixels for a StreamingBench video, so this release preserves 4.13.0.92. Do not silently downgrade OpenCV or upgrade NumPy when reproducing this setup.
2. **Decord 0.6.0's Linux wheel filename uses `py3-none-manylinux2010_x86_64`, but its internal WHEEL metadata says `cp36-cp36m-manylinux2010_x86_64`.** pip accepts the filename for installation on Python 3.10, then `pip check` reports an unsupported platform. The packaged Python/ctypes interface and video decoding were checked in the evaluated Python 3.10 environment. This release keeps the official wheel unchanged.

These two diagnostics are documented exceptions, not a claim that every `pip check` error is harmless. Investigate any additional dependency failures. A future change to this version combination requires separate media and model validation.

## Validation boundary

Versions were checked against the actual evaluation environment. Isolated media probes cover nine inputs from the eight benchmarks, including frame decoding, seeking, channel conversion and resizing. The selected OpenCV 4.13 and Decord 0.6 official wheels reproduce the checked baseline operations. Migration validation uses the existing Python/CUDA environment, CPU preflight and saved prediction/judge-cache replay. A complete fresh CUDA/flash-attn build and full benchmark inference are outside this validation.
