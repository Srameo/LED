from setuptools import setup, find_packages

setup(
    name='mipi_fewshot_raw_image_denoising',
    version='1.0.0',
    description='A scoring program for Few-shot RAW Image Denoising in MIPI challenge',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'skimage',
    ],
)