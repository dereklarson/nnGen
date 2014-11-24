import os
from setuptools import find_packages
from setuptools import setup

version = '0.01'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

install_requires = [
    'numpy',
    'Theano',
    ]

tests_require = [
    'mock',
    'pytest',
    ]

setup(
    name="nnGen",
    version=version,
    description="neural network generator",
    long_description="\n\n".join(README),
    keywords="",
    author="Derek Larson",
    author_email="larson.derek.a@gmail.com",
    url="https://github.com/dereklarson/nnGen",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require
        }
    )
