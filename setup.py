from setuptools import setup, find_packages

setup(
    name='t2l',
    version='1.0',
    author='Travis Chen',
    author_email='cxqlove1999@outlook.com',
    description='A package of machine learning used',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'IPython',
        'pandas',
    ]
)
