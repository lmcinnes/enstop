from setuptools import setup


def readme():
    with open("README.rst") as readme_file:
        return readme_file.read()


configuration = {
    "name": "enstop",
    "version": "0.2.2",
    "description": "Ensemble topic modelling with pLSA",
    "long_description": readme(),
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    "keywords": "topic model, LDA, pLSA, NMF",
    "url": "http://github.com/lmcinnes/enstop",
    "author": "Leland McInnes",
    "author_email": "leland.mcinnes@gmail.com",
    "maintainer": "Leland McInnes",
    "maintainer_email": "leland.mcinnes@gmail.com",
    "license": "BSD",
    "packages": ["enstop"],
    "install_requires": [
        "scikit-learn >= 0.23",
        "scipy >= 1.0",
        "numba >= 0.48",
        "dask[delayed] >= 1.2",
        "hdbscan >= 0.8",
        "umap-learn >= 0.3.8",
    ],
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "nose.collector",
    "tests_require": ["nose"],
    "data_files": (),
    "zip_safe": True,
}

setup(**configuration)
