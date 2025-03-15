#!/usr/bin/env python

import os
from setuptools import setup, find_packages

# Allow installation without pyproject.toml for compatibility with older tools
# Most modern tools will use pyproject.toml, but this serves as a fallback
if __name__ == "__main__":
    setup(
        name="llm-drift-detector",
        version="0.1.0",
        description="Monitoring and detection of drift in Large Language Models",
        author="Your Name",
        author_email="your.email@example.com",
        url="https://github.com/yourusername/llm-drift-detector",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.20.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "scipy>=1.7.0",
            "pyyaml>=6.0",
        ],
        extras_require={
            "openai": ["openai>=1.0.0"],
            "anthropic": ["anthropic>=0.3.0"],
            "huggingface": [
                "transformers>=4.25.0",
                "torch>=1.12.0",
                "sentence-transformers>=2.2.0",
            ],
            "vllm": ["vllm>=0.1.0"],
            "redis": ["redis>=4.5.0"],
            "sql": ["sqlalchemy>=2.0.0"],
            "visualization": [
                "dash>=2.8.0",
                "plotly>=5.13.0",
            ],
            "embeddings": ["sentence-transformers>=2.2.0"],
            "all": [
                "openai>=1.0.0",
                "anthropic>=0.3.0",
                "transformers>=4.25.0",
                "torch>=1.12.0",
                "sentence-transformers>=2.2.0",
                "vllm>=0.1.0",
                "redis>=4.5.0",
                "sqlalchemy>=2.0.0",
                "dash>=2.8.0",
                "plotly>=5.13.0",
            ],
            "dev": [
                "pytest>=7.0.0",
                "pytest-cov>=4.0.0",
                "black>=23.0.0",
                "isort>=5.0.0",
                "mypy>=1.0.0",
                "flake8>=6.0.0",
            ],
        },
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        keywords=[
            "llm",
            "drift",
            "monitoring",
            "machine learning",
            "artificial intelligence",
        ],
        long_description=open(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
            encoding="utf-8",
        ).read(),
        long_description_content_type="text/markdown",
    )