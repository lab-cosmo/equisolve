# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile

import equistore
import numpy as np
from equistore import TensorBlock, TensorMap


def array_from_block(block: TensorBlock) -> np.ndarray:
    """Extract parts of a :class:`equistore.TensorBlock` into a array.

    All components will be stacked along the rows / "sample"-dimension.
    This means that the number of coulums / "property"-dimension is unchanged.

    :param block:
        :class:`equistore.TensorBlock` for the extraction
    :returns M:
        :class:`numpy.ndarray` of shape (n, m) where m is the number of properties in
        the block.

    Note
    ----
    This function is used for creating the X and the y for Linear models. It may not
    work for Kernel models.
    """
    M = [block.values.reshape(-1, block.values.shape[-1])]
    for parameter in block.gradients_list():
        values = block.gradient(parameter).values
        M.append(values.reshape(-1, values.shape[-1]))
    return np.vstack(M)


def tensor_map_to_dict(tensor_map: TensorMap):
    """Format an object of a :class:`equistore.TensorBlock` into a dict of array.

    TODO rm usage of this function as soon
         https://github.com/lab-cosmo/equistore/issues/94
         is merged

    :param tensor_map:
        :class:`equistore.TensorMap` for the transform

    :returns tensor_map_dict:
        :class:`dict` of :class:`numpy.ndarray`, consistent with equistore.io.save
        format
    """
    tmp_filename = tempfile.mktemp() + ".npz"
    equistore.save(tmp_filename, tensor_map)
    tensor_map_dict = {key: value for key, value in np.load(tmp_filename).items()}
    os.remove(tmp_filename)
    return tensor_map_dict


def dict_to_tensor_map(tensor_map_dict: dict):
    """Format a dict of arrays complying with :class:`equistore.TensorBlock`

    TODO rm usage of this function as soon
         https://github.com/lab-cosmo/equistore/issues/94
         is merged

    :param tensor_map:
        :class:`dict` of :class:`numpy.ndarray`,
        consistent with :func:`equistore.save` format

    :returns tensor_map_dict:
        :class:`equistore.TensorMap` for the transform
    """
    tmp_filename = tempfile.mktemp() + ".npz"
    np.savez(tmp_filename, **tensor_map_dict)
    return equistore.load(tmp_filename)
