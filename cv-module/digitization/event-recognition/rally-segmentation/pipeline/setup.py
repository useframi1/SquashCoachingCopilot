"""Setup configuration for rally_state_pipeline package."""

from setuptools import setup, find_packages

setup(
    name="rally_state_pipeline",
    version="0.1.12",
    description="A package for detecting rally states in squash videos",
    author="Youssef Elhagg",
    author_email="yousseframi@aucegypt.edu",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "rally_state_pipeline": [
            "config.json",
            "models/ml/weights/*.pkl",
        ],
    },
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
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
