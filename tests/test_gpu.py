from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from face_finder import core

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not (Path("/dev/dri").is_dir() and any(Path("/dev/dri").iterdir())),
        reason="no DRI device on this host",
    ),
]

N_IMAGES = 8
WIDTH = 640
HEIGHT = 480


def _write_synthetic_image(path: Path, width: int = 640, height: int = 480, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


@pytest.fixture(scope="module")
def mp_images() -> list[mp.Image]:
    rng = np.random.default_rng(42)
    rgb = [rng.integers(0, 255, size=(HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(N_IMAGES)]
    return [mp.Image(image_format=mp.ImageFormat.SRGB, data=img) for img in rgb]


@pytest.fixture(scope="module")
def cpu_detector() -> mp.tasks.vision.FaceDetector:
    with core._create_detector(core._detector_options("cpu")) as detector:
        yield detector


@pytest.fixture(scope="module")
def gpu_detector() -> mp.tasks.vision.FaceDetector:
    # Intentionally no fallback: benchmark must fail loudly when the GPU
    # delegate cannot be created, otherwise numbers would silently be CPU.
    with core._create_detector(core._detector_options("gpu")) as detector:
        yield detector


def test_process_folder_smoke_in_gpu_container(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_image(src / "noise.png")
    out = tmp_path / "out"

    found = core.process_folder(src, out)

    assert out.exists()
    assert found == 0


def test_process_folder_gpu_delegate_smoke_in_gpu_container(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_image(src / "noise.png")
    out = tmp_path / "out"

    found = core.process_folder(src, out, device="gpu")

    assert out.exists()
    assert found == 0


def _detect_all(detector: mp.tasks.vision.FaceDetector, images: list[mp.Image]) -> None:
    for image in images:
        detector.detect(image)


def test_benchmark_detect_cpu(
    cpu_detector: mp.tasks.vision.FaceDetector, mp_images: list[mp.Image], benchmark: BenchmarkFixture
) -> None:
    benchmark(_detect_all, cpu_detector, mp_images)


def test_benchmark_detect_gpu(
    gpu_detector: mp.tasks.vision.FaceDetector, mp_images: list[mp.Image], benchmark: BenchmarkFixture
) -> None:
    benchmark(_detect_all, gpu_detector, mp_images)
