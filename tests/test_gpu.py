from pathlib import Path

import cv2
import numpy as np
import pytest

from face_finder import core

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not (Path("/dev/dri").is_dir() and any(Path("/dev/dri").iterdir())),
        reason="no DRI device on this host",
    ),
]


def _write_synthetic_image(path: Path, width: int = 640, height: int = 480, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_process_folder_smoke_in_gpu_container(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_image(src / "noise.png")
    out = tmp_path / "out"

    found = core.process_folder(src, out)

    assert out.exists()
    assert found == 0
