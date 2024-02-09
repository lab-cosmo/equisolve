# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

__version__ = "0.0.0-dev"
__authors__ = "the equisolve development team"


def refresh_global_flags():
    """
    Refreshes all global flags set on import of library. This function might be useful
    if one is in an interactive session and installed some of the optional dependenicies
    (torch, metatensor-torch) after importing the library.
    """
    global HAS_TORCH
    global HAS_METATENSOR_TORCH

    try:
        import torch  # noqa: F401

        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    try:
        from metatensor.torch import Labels, TensorBlock, TensorMap  # noqa: F401

        HAS_METATENSOR_TORCH = True
    except ImportError:
        from metatensor import Labels, TensorBlock, TensorMap  # noqa: F401

        HAS_METATENSOR_TORCH = False


# For a global consistent state of the package, we set the global flags once here
refresh_global_flags()
