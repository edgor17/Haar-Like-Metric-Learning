#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:58:53 2023

@author: Evan
"""
from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name='Haar-Like-Metric-Learning-testing',
    url='https://github.com/edgor17/Haar-Like-Metric-Learning-testing',
    author='Evan Gorman',
    author_email='evan.gorman@colorado.edu',
    packages=['AdaptiveHaarLike'],
    install_requires=['numpy','pandas','ete3','scipy','scikit-learn','tensorflow>=2.4.0','statsmodel'],
    version='0.1',
    description='Adaptive Metric Learning for Metagenomics - testing',
    long_description=long_description
)
