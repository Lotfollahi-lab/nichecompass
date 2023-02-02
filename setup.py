#!/usr/bin/env python

# This is a shim to hopefully allow Github to detect the package, build is done with poetry

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="autotalker",
        version='0.1.0',
        description="",
        author="Sebastian Birk",
        author_email="sebastian.birk@outlook.com",
        url="https://github.com/sebastianbirk/autotalker",
        packages=["autotalker"])
