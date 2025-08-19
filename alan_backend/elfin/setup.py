#!/usr/bin/env python3
"""
Setup script for ELFIN package.

This script installs the ELFIN package and its dependencies.
"""

from setuptools import setup, find_packages

setup(
    name="elfin",
    version="0.2.0",
    description="Embedded Language For Integrated Networks",
    author="ALAN Team",
    packages=find_packages(),
    install_requires=[
        "lark>=1.0.0",
        "pygls>=1.0.0",   # Language Server Protocol
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "mypy>=0.800",
        ],
        "lsp": [
            "pygls>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "elfin=alan_backend.elfin.cli:main",
            "elfin-lsp=alan_backend.elfin.lsp.server:start_server",
        ],
    },
)
