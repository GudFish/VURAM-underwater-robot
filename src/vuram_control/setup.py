#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='vuram_control',
    version='0.0.0',
    packages=find_packages('scripts'),
    package_dir={'': 'scripts'},
)
