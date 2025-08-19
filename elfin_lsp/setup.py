#!/usr/bin/env python3
"""
Setup script for the ELFIN Language Server.
"""

from setuptools import setup, find_packages

setup(
    name="elfin-lsp",
    version="0.1.0",
    author="ELFIN Team",
    author_email="elfin@example.com",
    description="Language Server Protocol implementation for the ELFIN language",
    long_description=open("README.md", encoding="utf-8").read() if isinstance(__file__, str) else "",
    long_description_content_type="text/markdown",
    url="https://github.com/elfin-team/elfin-lsp",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pygls>=1.0.0",
        "watchdog>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "elfin-lsp=elfin_lsp.cli:main",
        ],
    },
)
