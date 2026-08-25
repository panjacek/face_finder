#!/usr/bin/env python
import argparse
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import colorlog
import cv2
import mediapipe as mp
import numpy as np

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "blaze_face_full_range.tflite"

logger = logging.getLogger(__name__)


def config_logger(log_lvl: str) -> None:
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(log_color)s%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s]: %(message)s",
            log_colors={
                "DEBUG": "white",
                "INFO": "bold_white",
                "WARNING": "bold_yellow",
                "ERROR": "bold_red",
                "CRITICAL": "red,bg_yellow",
            },
            datefmt="%Y%m%d-%H:%M:%S",
            reset=True,
        )
    )
    logger.addHandler(handler)
    logger.setLevel(log_lvl)


def find_images(folder: str | Path) -> Iterator[Path]:
    root_path = Path(folder)
    if root_path.is_file():
        if root_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield root_path
        return
    for root, _dirs, files in os.walk(root_path):
        for file_name in files:
            path = Path(root) / file_name
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def _draw_detection(annotated_image: np.ndarray, detection: Any) -> None:
    # Tasks API returns pixel coords; the 6 keypoints are available on the detection
    # but intentionally not drawn (box + score is the chosen output style).
    box = detection.bounding_box
    x, y, w, h = int(box.origin_x), int(box.origin_y), int(box.width), int(box.height)
    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    score = detection.categories[0].score
    cv2.putText(
        annotated_image,
        f"{score:.2f}",
        (x, max(y - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def process_file(
    image_path: Path,
    face_detector: Any,  # mediapipe.tasks.vision.FaceDetector
    output_dir: Path,
) -> bool:
    t_start = time.monotonic()
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Failed to read {image_path}")
        return False
    t_after_read = time.monotonic()

    # MediaPipe Tasks expects SRGB; cv2 loads BGR.
    results = face_detector.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    )

    if not results.detections:
        logger.warning(f"Nothing found on {image_path}")
        return False

    annotated_image = image.copy()
    for detection in results.detections:
        _draw_detection(annotated_image, detection)

    out_path = output_dir / image_path.name
    if not cv2.imwrite(str(out_path), annotated_image):
        logger.warning(f"Failed to store output for {out_path}")
        return False

    t_done = time.monotonic()
    logger.debug(
        f"{image_path} read+detect in {round(1000 * (t_after_read - t_start), 4)}ms, "
        f"detection in {round(1000 * (t_done - t_after_read), 4)}ms"
    )
    logger.info(f"{image_path} found in {round(1000 * (t_done - t_start), 4)}ms")
    return True


def process_folder(folder: str | Path, output_dir: str | Path = "output") -> int:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Face detection model missing at {MODEL_PATH} - run 'make setup'")
    out_dir = Path(output_dir)
    if not out_dir.exists():
        logger.warning(f"Creating {out_dir}")
        out_dir.mkdir(parents=True)

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.7,
    )

    found = 0
    with mp.tasks.vision.FaceDetector.create_from_options(options) as face_detector:
        for image_path in find_images(folder):
            logger.debug(f"{image_path}")
            found += process_file(image_path, face_detector, out_dir)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="source dir or single file to parse")
    parser.add_argument("-l", "--log_lvl", default="INFO", help="logger lvl")
    parser.add_argument("-o", "--output_dir", default="output", help="dir for annotated images")
    args = parser.parse_args()
    config_logger(args.log_lvl)
    logger.info("=== Face Finder ===")
    process_folder(args.source, args.output_dir)


if __name__ == "__main__":
    main()
