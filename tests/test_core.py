from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
import pytest

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


def test_find_images_single_image_file(tmp_path: Path) -> None:
    img = tmp_path / "one.png"
    img.touch()
    assert list(core.find_images(img)) == [img]


def test_find_images_single_non_image_file(tmp_path: Path) -> None:
    txt = tmp_path / "one.txt"
    txt.touch()
    assert list(core.find_images(txt)) == []


def test_config_logger_sets_level() -> None:
    core.logger.handlers.clear()
    core.config_logger("DEBUG")
    assert core.logger.level == 10
    assert len(core.logger.handlers) == 1


def _write_synthetic_image(path: Path, width: int = 640, height: int = 480, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _fake_detector(detections: list) -> SimpleNamespace:
    return SimpleNamespace(detect=lambda image: SimpleNamespace(detections=detections))


def _fake_detection() -> list:
    return [
        SimpleNamespace(
            bounding_box=SimpleNamespace(origin_x=160, origin_y=120, width=320, height=240),
            categories=[SimpleNamespace(score=0.9)],
        )
    ]


def test_process_file_writes_annotated_output(tmp_path: Path) -> None:
    src = tmp_path / "face.png"
    _write_synthetic_image(src)
    out = tmp_path / "out"
    out.mkdir()

    ok = core.process_file(src, _fake_detector(_fake_detection()), out)

    assert ok is True
    assert (out / "face.png").exists()


def test_process_file_no_detections_returns_false(tmp_path: Path) -> None:
    src = tmp_path / "noise.png"
    _write_synthetic_image(src)
    out = tmp_path / "out"

    ok = core.process_file(src, _fake_detector([]), out)

    assert ok is False
    assert not out.exists()


def test_process_file_unreadable_image_returns_false(tmp_path: Path) -> None:
    src = tmp_path / "broken.png"
    src.write_bytes(b"not an image")

    ok = core.process_file(src, _fake_detector(_fake_detection()), tmp_path / "out")

    assert ok is False


def test_process_file_write_failure_returns_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "face.png"
    _write_synthetic_image(src)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(core.cv2, "imwrite", lambda *args: False)

    ok = core.process_file(src, _fake_detector(_fake_detection()), out)

    assert ok is False


def test_process_folder_missing_model_exits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "MODEL_PATH", tmp_path / "nope.tflite")

    with pytest.raises(SystemExit, match="make setup"):
        core.process_folder(tmp_path)


def test_process_folder_smoke_real_detector(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_image(src / "noise.png")
    out = tmp_path / "out"

    found = core.process_folder(src, out)

    assert out.exists()
    assert found == 0


def test_detector_options_delegate_per_device() -> None:
    cpu_options = core._detector_options("cpu")
    gpu_options = core._detector_options("gpu")

    assert cpu_options.base_options.delegate == mp.tasks.BaseOptions.Delegate.CPU
    assert gpu_options.base_options.delegate == mp.tasks.BaseOptions.Delegate.GPU
    assert str(gpu_options.base_options.model_asset_path).endswith("blaze_face_full_range.tflite")
    assert gpu_options.running_mode == mp.tasks.vision.RunningMode.IMAGE


class _CtxDetector:
    def __init__(self, detections: list) -> None:
        self._inner = _fake_detector(detections)

    def detect(self, image: object) -> SimpleNamespace:
        return self._inner.detect(image)

    def __enter__(self) -> _CtxDetector:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_process_folder_gpu_unavailable_falls_back_to_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen_delegates: list = []

    def fake_create(options: SimpleNamespace) -> _CtxDetector:
        delegate = options.base_options.delegate
        if not seen_delegates:
            seen_delegates.append(delegate)
            raise RuntimeError("no EGL")
        seen_delegates.append(delegate)
        return _CtxDetector([])

    monkeypatch.setattr(core, "_create_detector", fake_create)
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"

    found = core.process_folder(src, out, device="gpu")

    assert found == 0
    assert seen_delegates == [
        mp.tasks.BaseOptions.Delegate.GPU,
        mp.tasks.BaseOptions.Delegate.CPU,
    ]


def test_process_folder_cpu_failure_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_create(options: SimpleNamespace) -> _CtxDetector:
        raise RuntimeError("model broken")

    monkeypatch.setattr(core, "_create_detector", fake_create)

    with pytest.raises(RuntimeError, match="model broken"):
        core.process_folder(tmp_path, device="cpu")
