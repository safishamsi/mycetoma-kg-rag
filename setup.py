from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh 
        if line.strip() and not line.startswith("#") and not line.startswith("python>=")
    ]

setup(
    name="mycetoma-kg-rag",
    version="1.0.0",
    author="First Author, Second Author, Third Author",
    author_email="first.author@university.edu",
    description="Multi-Modal Knowledge Graph-Augmented Retrieval for Explainable Mycetoma Diagnosis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mycetoma-kg-rag",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/mycetoma-kg-rag/issues",
        "Documentation": "https://yourusername.github.io/mycetoma-kg-rag",
        "Source Code": "https://github.com/yourusername/mycetoma-kg-rag",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
        ],
        "gpu": [
            "torch>=1.12.0+cu116",
            "torchvision>=0.13.0+cu116",
        ],
        "biomedical": [
            "biopython>=1.79",
        ],
    },
    entry_points={
        "console_scripts": [
            "mycetoma-diagnose=src.pipeline.diagnostic_system:main",
            "mycetoma-train=scripts.train_inception_v3:main",
            "mycetoma-build-kg=scripts.build_kg:main",
            "mycetoma-evaluate=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
)
