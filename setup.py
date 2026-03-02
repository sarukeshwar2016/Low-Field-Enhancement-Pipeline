"""setup.py -- installable package for the Low-Field MRI Enhancement Pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name             = "lowfield-mri-pipeline",
    version          = "3.0.0",
    author           = "MRI Enhancement Research Team",
    description      = "Physics-driven low-field MRI simulation and enhancement",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = requirements,
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    entry_points = {
        "console_scripts": [
            "lf-simulate=dicom_to_lf_sim:main",
            "lf-enhance=enhanced_batch_9_100:main",
        ],
    },
)
