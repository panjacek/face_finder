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

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

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
    for root, _dirs, files in os.walk(folder):
        for file_name in files:
            path = Path(root) / file_name
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path


def process_file(
    image_path: Path,
    face_detection: Any,  # mediapipe.solutions.face_detection.FaceDetection
    output_dir: Path,
) -> bool:
    mp_drawing = mp.solutions.drawing_utils
    t_start = time.monotonic()
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Failed to read {image_path}")
        return False
    t_after_read = time.monotonic()

    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.detections:
        logger.warning(f"Nothing found on {image_path}")
        return False

    annotated_image = image.copy()
    for detection in results.detections:
        mp_drawing.draw_detection(annotated_image, detection)

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
    mp_face_detection = mp.solutions.face_detection
    out_dir = Path(output_dir)
    if not out_dir.exists():
        logger.warning(f"Creating {out_dir}")
        out_dir.mkdir(parents=True)

    found = 0
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        for image_path in find_images(folder):
            logger.debug(f"{image_path}")
            found += process_file(image_path, face_detection, out_dir)
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
