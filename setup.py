from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="geniusrise-ocr",
    version="0.1.7",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=requirements,
    python_requires=">=3.10",
    author="ixaxaar",
    author_email="ixaxaar@geniusrise.ai",
    description="ocr bolts for geniusrise",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geniusrise/geniusrise-ocr",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="mlops, llm, geniusrise, machine learning, data processing",
    project_urls={
        "Bug Reports": "https://github.com/geniusrise/geniusrise-ocr/issues",
        "Source": "https://github.com/geniusrise/geniusrise-ocr",
        "Documentation": "https://docs.geniusrise.ai/",
    },
    package_data={
        "geniusrise": [],
    },
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage"],
    },
)
