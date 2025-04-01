"""
Deep OrderBook - Advanced cryptocurrency order book analysis toolkit
"""
import runpy
from setuptools import setup, find_packages

PACKAGE_NAME = "deep_orderbook"
version_meta = runpy.run_path("./version.py")
VERSION = version_meta["__version__"]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def parse_requirements(requirements_file: str = 'requirements.txt') -> list[str]:
    """Get the contents of a file listing the requirements"""
    with open(requirements_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    dependencies = []
    for line in lines:
        maybe_dep = line.strip()
        if maybe_dep.startswith('#') or not maybe_dep:
            # Skip comment lines or empty lines
            continue
        if maybe_dep.startswith('git+'):
            # VCS reference for dev purposes, expect a trailing comment
            # with the normal requirement
            __, __, maybe_dep = maybe_dep.rpartition('#')
        else:
            # Ignore any trailing comment
            maybe_dep, __, __ = maybe_dep.partition('#')
        # Remove any whitespace and assume non-empty results are dependencies
        maybe_dep = maybe_dep.strip()
        if maybe_dep:
            dependencies.append(maybe_dep)
    return dependencies


if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description="Transforms order book data into temporally and spatially local-correlated representations for quantitative analysis and deep learning",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/gQuantCoder/deep_orderbook",
        author="gQuantCoder",
        author_email="gquantcoder@gmail.com",
        packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
        include_package_data=True,
        install_requires=parse_requirements("requirements.txt"),
        python_requires=">=3.12",
        entry_points={
            "console_scripts": [
                "deepbook=deep_orderbook.__main__:main",
                "deepbook-record=deep_orderbook.consumers.recorder:main",
                "deepbook-replay=deep_orderbook.replayer:main"
            ]
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Financial and Insurance Industry",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Office/Business :: Financial :: Investment",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        keywords="orderbook, trading, cryptocurrency, machine learning, finance",
    )