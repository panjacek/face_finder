import hashlib
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
if not os.path.isdir(ROOT_DIR):
    # /app only exists inside the container; fall back to cwd when running locally.
    ROOT_DIR = os.getcwd()
OUTPUT_DIR = os.environ.get("FACE_FINDER_OUTPUT", str(Path(ROOT_DIR) / "output"))
DEVICE = os.environ.get("FACE_FINDER_DEVICE", "cpu")
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


folder_options = [
    x[0]
    for x in os.walk(ROOT_DIR)
    if not any(part.startswith(".") or part in IGNORED_FOLDERS for part in Path(x[0]).parts[1:])
    and not (x[0] + os.sep).startswith(OUTPUT_DIR + os.sep)
]

if not folder_options:
    st.warning(f"No browsable folders under {ROOT_DIR}")
    folder_selected = None
else:
    folder_selected = st.selectbox(
        "folder_to_show",
        folder_options,
        index=None,
    )


def _output_subdir(folder: str) -> str:
    tag = hashlib.md5(folder.encode(), usedforsecurity=False).hexdigest()[:8]
    return str(Path(OUTPUT_DIR) / f"{Path(folder).name}_{tag}")


def _folder_signature(folder: str) -> tuple[tuple[str, int, int], ...]:
    # Content fingerprint so the detection cache invalidates when folder contents change.
    return tuple((path, os.stat(path).st_mtime_ns, os.stat(path).st_size) for path in get_images(folder))


@st.cache_data(show_spinner="Detecting faces...")
def detect_faces(folder: str, out_dir: str, signature: tuple) -> list[str]:
    core.process_folder(folder, out_dir, DEVICE)
    return get_images(out_dir)


def display_folder(folder_selected: str, show_faces: bool) -> None:
    folder = Path(folder_selected)
    if show_faces:
        image_list = detect_faces(str(folder), _output_subdir(str(folder)), _folder_signature(str(folder)))
    else:
        image_list = get_images(str(folder))

    if len(image_list) == 0:
        st.warning("No images found in selected folder")
        return

    view = "faces" if show_faces else "raw"
    thumbs_paginator = paginator(
        f"Photos to check ({view})",
        image_list,
        items_per_page=30,
        on_sidebar=True,
        key=f"pag_{view}",
    )

    _indices_on_page, images_on_page = map(list, zip(*thumbs_paginator))

    st.image(images_on_page, width=300)


if folder_selected:
    view = st.radio("View", ["Raw", "Faces"], index=None, horizontal=True, key="view_mode")
    if view == "Raw":
        display_folder(folder_selected, False)
    elif view == "Faces":
        display_folder(folder_selected, True)
