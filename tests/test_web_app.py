from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "src" / "web_app" / "app.py"


def _write_noise_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


@pytest.fixture
def image_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_noise_image(tmp_path / "one.png", 7)
    _write_noise_image(tmp_path / "two.jpg", 8)
    monkeypatch.setenv("FACE_FINDER_ROOT", str(tmp_path))
    monkeypatch.setenv("FACE_FINDER_OUTPUT", str(tmp_path / "out"))
    return tmp_path


@pytest.fixture
def empty_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FACE_FINDER_ROOT", str(tmp_path))
    monkeypatch.setenv("FACE_FINDER_OUTPUT", str(tmp_path / "out"))
    return tmp_path


def _boot() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_app_boots_without_folder_selection(image_root: Path) -> None:
    _boot()


def test_app_selecting_folder_shows_checkboxes(image_root: Path) -> None:
    at = _boot()
    at.selectbox[0].set_value(str(image_root)).run()

    assert not at.exception
    labels = [cb.label for cb in at.checkbox]
    assert labels == ["Show Raw", "Show Faces"]


def test_app_show_faces_creates_output_dir(image_root: Path) -> None:
    at = _boot()
    at.selectbox[0].set_value(str(image_root)).run()
    at.checkbox[1].check().run(timeout=120)

    assert not at.exception
    assert (image_root / "out").is_dir()


def test_app_empty_folder_warns(empty_root: Path) -> None:
    at = _boot()
    at.selectbox[0].set_value(str(empty_root)).run()
    at.checkbox[0].check().run()

    assert not at.exception
    assert len(at.warning) == 1


def test_app_show_raw_renders_images(image_root: Path) -> None:
    at = _boot()
    at.selectbox[0].set_value(str(image_root)).run()
    at.checkbox[0].check().run()

    assert not at.exception
    assert len(at.image) == 1
    assert len(at.image[0].value) == 2
