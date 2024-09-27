#!/usr/bin/env python
import argparse
import logging
import os
import time

import colorlog
import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


def config_logger(log_lvl):
    global logger
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
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(log_lvl)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="source dir or single file to parse")
    parser.add_argument("-l", "--log_lvl", default="INFO", help="logger lvl")
    args = parser.parse_args()
    config_logger(args.log_lvl)
    logger.info("=== Face Finder ===")
    process_folder(args.source)


def process_folder(folder: str) -> None:
    mp_face_detection = mp.solutions.face_detection

    output_dir = "output"
    if not os.path.exists(output_dir):
        logger.warning(f"Creating {output_dir}")
        os.mkdir(output_dir)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face_detection:
        for root, dirs, files in os.walk(folder):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                logger.debug(f"{file_path}")
                if not any(x in file_name for x in ["jpg", "png"]):
                    continue

                process_file(file_path, file_name, face_detection, output_dir)


def process_file(file_path, file_name, face_detection, output_dir) -> None:
    mp_drawing = mp.solutions.drawing_utils
    t_start = time.monotonic()
    image = cv2.imread(file_path)
    t_after_read_start = time.monotonic()
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    t_post_detection = time.monotonic()

    # Draw face detections of each face.
    if not results.detections:
        logger.warning(f"Nothing found on {file_path}")
        return

    annotated_image = image.copy()
    for detection in results.detections:
        mp_drawing.draw_detection(annotated_image, detection)

    if not cv2.imwrite(os.path.join(output_dir, file_name), annotated_image):
        logger.warning(f"Failed to store output for {file_name}")

    logger.debug(
        f"{file_path} detection with read:"
        f"in {round(1000 * (t_post_detection - t_start), 4)}ms, "
        f"detection in {round(1000 * (t_post_detection - t_after_read_start), 4)}ms"
    )
    logger.info(f"{file_path} found in {round(1000*(time.monotonic() - t_start), 4)}ms")


if __name__ == "__main__":
    main()
