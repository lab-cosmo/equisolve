# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
"""Classes to build models from an class:`metatensor.TensorMap`.

Model classes listed here use are based on Numpy. Classes based
on torch will be added in the future.
"""


from .linear_model import Ridge
from .sor_kernel_ridge import SorKernelRidge


__all__ = ["Ridge", "SorKernelRidge"]
