import os
from pathlib import Path

import streamlit as st
from streamlit.logger import get_logger

from face_finder import face_finder
from helpers.paginator import paginator

st.set_page_config(page_title="Face Finder", page_icon=":camera:", layout="wide")

# add sublogger
logger = get_logger(__name__)
logger.setLevel("INFO")


st.title("Face Finder")

# select folder to show
st.write("Select a folder")


IGNORED_FOLDERS = [".git", "__pycache__"]
FILE_TYPES_TO_SHOW = [".jpg", ".png"]
OUTPUT_DIR = "/app/output"


def get_images(folder):
    image_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(file_type) for file_type in FILE_TYPES_TO_SHOW):
                image_list.append(os.path.join(root, file))
    return image_list


folder_selected = st.selectbox(
    "folder_to_show",
    [x[0] for x in os.walk("/app") if not any(ignored_x in x[0] for ignored_x in IGNORED_FOLDERS)],
    index=None,
)


def display_folder(folder_selected, show_faces):
    folder = Path(folder_selected)
    image_list = []
    if show_faces:
        face_finder.process_folder(folder)
        image_list = get_images(OUTPUT_DIR)
    else:
        image_list = get_images(folder)

    if len(image_list) == 0:
        st.warning("No images found in selected folder")
        return

    thumbs_paginator = paginator(
        "Photos to check",
        image_list,
        items_per_page=30,
        on_sidebar=True,
    )

    indices_on_page, images_on_page = map(list, zip(*thumbs_paginator))
    # captions_to_show = image_list

    st.image(images_on_page, width=300)


if folder_selected:
    if st.checkbox("Show Raw", value=False):
        display_folder(folder_selected, False)
    show_faces = st.checkbox("Show Faces", value=False)

    if show_faces:
        display_folder(folder_selected, show_faces)
