"""Benchmark detect-only latency of the face detector per device.

Preloads images into RAM so timing covers MediaPipe inference only (no disk
I/O). Runs CPU first, then GPU; GPU failures fall back inside
core.process_folder logic is NOT used here - a failed GPU delegate aborts the
GPU section with a warning so the CPU numbers still print.

Usage: uv run python scripts/bench_device.py path/to/images [-n 50]
"""

import argparse
import statistics
import time

import cv2
import mediapipe as mp

from face_finder.core import MODEL_PATH, _detector_options, find_images


def _load_images(folder: str) -> list:
    images = []
    for path in find_images(folder):
        img = cv2.imread(str(path))
        if img is None:
            print(f"skipping unreadable {path}")
            continue
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images


def _bench(images: list, device: str, repeats: int) -> list[float]:
    options = _detector_options(device)
    latencies_ms: list[float] = []
    with mp.tasks.vision.FaceDetector.create_from_options(options) as detector:
        for _ in range(repeats):
            for rgb in images:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                t0 = time.monotonic()
                detector.detect(mp_image)
                latencies_ms.append(1000 * (time.monotonic() - t0))
    return latencies_ms


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Face detection model missing at {MODEL_PATH} - run 'make setup'")
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="dir or single image to benchmark")
    parser.add_argument("-n", "--repeats", type=int, default=1, help="passes over the image set [1]")
    parser.add_argument("--skip-gpu", action="store_true", help="benchmark CPU only")
    args = parser.parse_args()

    images = _load_images(args.source)
    if not images:
        raise SystemExit(f"no readable images under {args.source}")
    print(f"{len(images)} images x {args.repeats} repeat(s)\n")

    devices = ["cpu"] if args.skip_gpu else ["cpu", "gpu"]
    for device in devices:
        try:
            latencies = _bench(images, device, args.repeats)
        except Exception as exc:
            print(f"{device.upper()}: unavailable ({exc})\n")
            continue
        n = len(latencies)
        total_s = sum(latencies) / 1000
        print(
            f"{device.upper()}: n={n} "
            f"median={statistics.median(latencies):.2f}ms "
            f"mean={statistics.fmean(latencies):.2f}ms "
            f"p95={sorted(latencies)[int(0.95 * (n - 1))]:.2f}ms "
            f"throughput={n / total_s:.1f} fps"
        )


if __name__ == "__main__":
    main()
