from setuptools import setup, find_packages

setup(
    name='GANsforVirtualEye',
    version='0.1.0',
    author='Shailendra Bhandari',
    author_email='shailendra.bhandari@oslomet.no',
    description='This package provides an implementation of Generative Adversarial Networks (GANs) for time series generation, with flexible architecture options. Users can select different combinations of generator and discriminator models, including Convolutional Neural Networks (CNN) and Long Short-Term Memory networks (LSTM), to suit their specific needs.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'matplotlib',
        'scipy',
        'sklearn',
        'pandas',
        'progressbar2'
    ],
    python_requires='>=3.6',
)

