from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of your README.md file
with open(HERE / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="GANsforVirtualEye",
    version="0.1.5",
    author="Shailendra Bhandari",
    author_email="shailendra.bhandari@oslomet.no",
    description=(
        "This package provides an implementation of Generative Adversarial Networks (GANs) "
        "for time series generation, with flexible architecture options. Users can select "
        "different combinations of generator and discriminator models, including Convolutional "
        "Neural Networks (CNN) and Long Short-Term Memory networks (LSTM), to suit their specific needs."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shailendrabhandari/GANsForVirtualEye.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "pandas",
        "progressbar2",
    ],
)
