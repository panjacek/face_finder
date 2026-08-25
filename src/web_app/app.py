import os
from pathlib import Path

import streamlit as st
from streamlit.logger import get_logger

from face_finder import core
from face_finder.paginator import paginator

st.set_page_config(page_title="Face Finder", page_icon=":camera:", layout="wide")

logger = get_logger(__name__)
logger.setLevel("INFO")

ROOT_DIR = os.environ.get("FACE_FINDER_ROOT", "/app")
OUTPUT_DIR = os.environ.get("FACE_FINDER_OUTPUT", str(Path(ROOT_DIR) / "output"))
IGNORED_FOLDERS = [".git", "__pycache__"]
FILE_TYPES_TO_SHOW = [".jpg", ".jpeg", ".png"]

st.title("Face Finder")

# select folder to show
st.write("Select a folder")


def get_images(folder: str) -> list[str]:
    image_list = []
    for root, _dirs, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(file_type) for file_type in FILE_TYPES_TO_SHOW):
                image_list.append(os.path.join(root, file))
    return sorted(image_list)


folder_selected = st.selectbox(
    "folder_to_show",
    [x[0] for x in os.walk(ROOT_DIR) if not any(ignored_x in x[0] for ignored_x in IGNORED_FOLDERS)],
    index=None,
)


def display_folder(folder_selected: str, show_faces: bool) -> None:
    folder = Path(folder_selected)
    if show_faces:
        core.process_folder(folder, OUTPUT_DIR)
        image_list = get_images(OUTPUT_DIR)
    else:
        image_list = get_images(str(folder))

    if len(image_list) == 0:
        st.warning("No images found in selected folder")
        return

    thumbs_paginator = paginator(
        "Photos to check",
        image_list,
        items_per_page=30,
        on_sidebar=True,
    )

    _indices_on_page, images_on_page = map(list, zip(*thumbs_paginator))

    st.image(images_on_page, width=300)


if folder_selected:
    if st.checkbox("Show Raw", value=False):
        display_folder(folder_selected, False)
    show_faces = st.checkbox("Show Faces", value=False)

    if show_faces:
        display_folder(folder_selected, show_faces)
