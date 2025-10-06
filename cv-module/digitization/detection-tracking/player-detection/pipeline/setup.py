"""Setup configuration for player_tracking_pipeline package."""

from setuptools import setup, find_packages

setup(
    name="player_tracking_pipeline",
    version="0.2.2",
    description="A package for detecting players in squash videos",
    author="Youssef Elhagg",
    author_email="yousseframi@aucegypt.edu",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "player_tracking_pipeline": [
            "config.json",
            "model/weights/*.pt",
        ],
    },
    install_requires=[
        "numpy==1.26.4",
        "opencv-python==4.10.0.84",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        "ultralytics==8.3.101",
        "scipy==1.12.0",
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
