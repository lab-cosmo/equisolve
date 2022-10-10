#!/usr/bin/env python3

# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
import re

from setuptools import setup


VERSION = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', open("src/equisolve/__init__.py").read()
).group(1)

setup(
    setuptools_git_versioning={
        "enabled": True,
        "starting_version": VERSION,
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}-{sha}",
        "dirty_template": "{tag}.dev{ccount}-{sha}+dirty",
    }
)
