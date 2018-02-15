#!/usr/bin/env python

from setuptools import setup, find_packages

def _read(fname):
    try:
        with open(fname) as fobj:
            return fobj.read()

    except IOError:
        return ''


setup(
    name='spline_utils',
    version='0.0.1',
    description='collection of functions for using splines for regression',
    long_description=_read("Readme.md"),
    author='Matthias Ossadnik',
    author_email='ossadnik.matthias@gmail.com',
    license='MIT',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url="https://github.com/mossadnik/spline_utils.git",
    setup_requires=['pytest-runner'],
    install_requires=[],
    tests_require=['pytest'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
)
