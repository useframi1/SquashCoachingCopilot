"""Setup configuration for ball_detection_pipeline package."""

from setuptools import setup, find_packages

setup(
    name="ball_detection_pipeline",
    version="0.1.6",
    description="A package for detecting the ball in squash videos",
    author="Youssef Elhagg",
    author_email="yousseframi@aucegypt.edu",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "ball_detection_pipeline": [
            "config.json",
            "models/tracknet/weights/*.pt",
        ],
    },
    install_requires=[
        "numpy",
        "opencv-python",
        "torch",
        "inference",
        "supervision",
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
