# Benchmarks

Detect-only latency for the face detector, CPU vs GPU delegate.
Working file — extended as new hosts/devices get tested.

## Methodology

- `scripts/bench_device.py <folder> -n 30` — images preloaded into RAM,
  timing covers MediaPipe `detect()` calls only (no disk I/O, no decode)
- `-n 30` = 30 passes over the image set
- container equivalent: `make test-gpu` runs the same detect-only timing
  as pytest-benchmark tests on synthetic images (`tests/test_gpu.py`),
  with min/median/IQR/outlier stats; the GPU benchmark has no CPU
  fallback — it fails loudly instead of mislabeling numbers
- GPU path = MediaPipe Tasks GPU delegate over EGL/GLES (Mesa radeonsi on
  AMD, NVIDIA EGL via glvnd ICD pin); CPU path = XNNPACK delegate
- noise in output that can be ignored: TensorFlow Lite delegate
  creation INFO lines, `inference_feedback_manager.cc` warning,
  `tensor.cc` E0000 spam, `symbolize_elf.inc` RAW warning

## Results

### 2026-08-26 — Ryzen 2400G desktop: Vega 11 iGPU + GT1030 dGPU

Set: `fake_set1` (15 images) × 30 passes → n=450 per device.

| device | median | mean | p95 | throughput |
| ------ | ------ | ---- | --- | ---------- |
| cpu (XNNPACK) | 11.17–14.82 ms¹ | — | 13.94–18.16 ms¹ | 68–88 fps |
| gpu (Vega 11, radeonsi EGL, Mesa 26.0.8) | 9.37 ms | 9.79 ms | 12.07 ms | 102 fps |
| gpu (GT1030, NVIDIA proprietary EGL, driver 580.173.02) | 6.29 ms | 6.67 ms | 8.04 ms | 150 fps |

¹ two runs on the same host differed by ~33% — CPU numbers are noisy,
treat as ballpark; GPU medians were stable.

### 2026-08-26 — same host, nvidia-container-toolkit container

Same `fake_set1` × 30. Numbers are **absolute different** from bare metal
(~5× slower on both CPU and GPU) due to containerised mediapipe build;
the **relative** 2.2× GPU speedup is consistent.

| device | median | throughput |
| ------ | ------ | ---------- |
| cpu (XNNPACK) | 79.8 ms | 12 fps |
| gpu (GT1030, nvidia-container-toolkit) | 36.9 ms | 27 fps |

Takeaways:

- GT1030 = **2.4× its own CPU** bare metal, **2.2×** in container — consistent
  relative gain across environments
- container absolute numbers ~5× slower than bare metal (mediapipe build
  difference, not hardware)
- iGPU vs CPU gain modest (~16%), as expected for a ~1 MB model
  (see docs/model_choice.md)
- device selection discovery: with
  `__EGL_VENDOR_LIBRARY_FILENAMES=...10_nvidia.json` pinned,
  mediapipe's default-display probe failed (`eglGetDisplay` error 0x3000)
  but its internal **multi-GPU EGL display selection** fallback found the
  NVIDIA anyway (`gl_context.cc:96 ... trying multi-gpu EGL display
  selection`). The delegate is more resilient than documented.

### Probe recipes (this host)

```bash
# Vega 11 (default EGL display lands on Mesa)
uv run python scripts/bench_device.py ./fake_set1 -n 30

# GT1030 (pin glvnd ICD; mediapipe multi-gpu fallback does the rest)
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
uv run python scripts/bench_device.py ./fake_set1 -n 30
```

## TODO / want-to-test

- bigger set (`DB_jpg`, FERET gallery) and larger resolutions
- `-n` scaling (warmup effects, first-call JIT of delegates)
- re-run CPU a few times to quantify run-to-run variance
- container path (`make test-gpu`) — confirm delegate actually creates
  under `/dev/dri` mount; add Mesa libs to Dockerfile if it falls back;
  NVIDIA compose variant would need nvidia-container-toolkit +
  `NVIDIA_DRIVER_CAPABILITIES=graphics` instead of the DRI mount
