from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from mediapipe.framework.formats import detection_pb2

from face_finder import core


@pytest.fixture
def image_tree(tmp_path: Path) -> Path:
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.JPG").touch()
    (tmp_path / "c.jpeg").touch()
    (tmp_path / "d.PNG").touch()
    (tmp_path / "e.txt").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "f.png").touch()
    return tmp_path


def test_find_images_filters_by_extension_case_insensitive(image_tree: Path) -> None:
    found = sorted(p.name for p in core.find_images(image_tree))
    assert found == ["a.jpg", "b.JPG", "c.jpeg", "d.PNG", "f.png"]


def test_find_images_empty_dir(tmp_path: Path) -> None:
    assert list(core.find_images(tmp_path)) == []


def test_config_logger_sets_level() -> None:
    core.logger.handlers.clear()
    core.config_logger("DEBUG")
    assert core.logger.level == 10
    assert len(core.logger.handlers) == 1


def _write_synthetic_image(path: Path, width: int = 640, height: int = 480, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


class FakeDetector:
    def __init__(self, detections: list) -> None:
        self._detections = detections

    def process(self, image: np.ndarray) -> SimpleNamespace:
        return SimpleNamespace(detections=self._detections)


def _fake_detection() -> list:
    d = detection_pb2.Detection()
    d.score.append(0.9)
    d.location_data.format = d.location_data.RELATIVE_BOUNDING_BOX
    box = d.location_data.relative_bounding_box
    box.xmin = 0.25
    box.ymin = 0.25
    box.width = 0.5
    box.height = 0.5
    return [d]


def test_process_file_writes_annotated_output(tmp_path: Path) -> None:
    src = tmp_path / "face.png"
    _write_synthetic_image(src)
    out = tmp_path / "out"
    out.mkdir()

    ok = core.process_file(src, FakeDetector(_fake_detection()), out)

    assert ok is True
    assert (out / "face.png").exists()


def test_process_file_no_detections_returns_false(tmp_path: Path) -> None:
    src = tmp_path / "noise.png"
    _write_synthetic_image(src)
    out = tmp_path / "out"

    ok = core.process_file(src, FakeDetector([]), out)

    assert ok is False
    assert not out.exists()


def test_process_folder_smoke_real_detector(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_image(src / "noise.png")
    out = tmp_path / "out"

    found = core.process_folder(src, out)

    assert out.exists()
    assert found == 0
