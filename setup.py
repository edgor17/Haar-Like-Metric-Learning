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
    name='Haar-Like-Metric-Learning',
    url='https://github.com/edgor17/Haar-Like-Metric-Learning',
    author='Evan Gorman',
    author_email='evan.gorman@colorado.edu',
    packages=['AdaptiveHaarLike'],
    install_requires=['numpy','pandas','ete3','scipy','matplotlib','scikit-learn','seaborn'],
    version='0.1',
    description='Adaptive Metric Learning for Metagenomics',
    long_description=long_description
)
