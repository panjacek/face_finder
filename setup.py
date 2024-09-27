import setuptools

setuptools.setup(
    name="face_finder",
    version="0.0.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["face_finder"],
    zip_safe=False,
    python_requires=">=3.12",
    entry_points={
        "console_scripts": [
            "face_finder=face_finder.face_finder:main",
        ],
    },
)
