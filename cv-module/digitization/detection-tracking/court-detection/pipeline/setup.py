"""Setup configuration for court_detection_pipeline package."""

from setuptools import setup, find_packages

setup(
    name="court_detection_pipeline",
    version="0.1.4",
    description="A package for detecting and calibrating the court in squash videos",
    author="Youssef Elhagg",
    author_email="yousseframi@aucegypt.edu",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "court_detection_pipeline": [
            "config.json",
            "model/weights/*.pt",
        ],
    },
    install_requires=[
        "numpy",
        "inference",
        "opencv-python",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
