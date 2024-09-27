from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="progressive-scheduling",
    version="0.1.0",
    author="Christian Wallenwein",
    description="A progress-based scheduling library for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cwallenwein/progressive-scheduling",
    project_urls={
        "Bug Tracker": "https://github.com/cwallenwein/progressive-scheduling/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.0",
    ],
    extras_require={
        "dev": ["black", "flake8", "isort"],
    },
)
