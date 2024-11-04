from setuptools import setup, find_packages

setup(
    name='gan_package',
    version='0.1.0',
    author='Shailendra Bhandari',
    author_email='shailendra.bhandari@oslomet.no',
    description='GAN for Time Series Generation',
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

