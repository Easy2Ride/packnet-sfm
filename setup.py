from setuptools import setup

# To install, cd to parent dir and run pip install -e <packnet_sfm_folder>

__version__ = "0.0.1"

setup(
    name='packnet_sfm',
    version=__version__,
    description='PackNet-Sfm fork',
    url='https://github.com/Easy2Ride/packnet-sfm',
    author='Easy2Ride',
    author_email='e2r.htz@gmail.com',
    keywords='deep learning',
    packages=[
        'packnet_sfm',
    ],
    license='MIT License',
)