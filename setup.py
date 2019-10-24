from setuptools import setup

install_requires = [
    'tqdm',
    'scipy>=1.1.0',
    'hetio==0.2.9',
    'scikit-learn>=0.19.2',
    'matplotlib>=2.2',
    'pandas',
    'numpy',
]

setup(
    name='hetnet_ml',
    author='Mike Mayers',
    author_email='mmayers@scripps.edu',
    url='https://github.com/mmayers12/hetnet_ml',
    version='0.0.1',
    packages=['hetnet_ml'],
    license='LICENSE',
    description='Matrix-based feature extraction for Hetnets',
    long_description=open('README.rst').read(),
    install_requires=install_requires,
    python_requires='>=3.6',
)
