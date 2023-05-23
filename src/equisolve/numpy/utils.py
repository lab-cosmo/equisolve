# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile
from typing import List

import numpy as np
from equistore import Labels, TensorBlock, TensorMap
from equistore.io import load, save


def block_to_array(block: TensorBlock, parameter_keys: List[str]) -> np.ndarray:
    """Extract parts of a :class:`equistore.TensorBlock` into a array.

    All components will be stacked along the rows / "sample"-dimension.
    This means that the number of coulums / "property"-dimension is unchanged.

    :param block:
        :class:`equistore.TensorBlock` for the extraction
    :param value_keys:
        List of keys specifying the parameter of the block which will be extracted.
    :returns M:
        :class:`numpy.ndarray` of shape (n, m) where m is the number of properties in
        the block.

    Note
    ----
    This function is used for creating the X and the y for Linear models. It may not
    work for Kernel models.
    """
    M = []
    for parameter in parameter_keys:
        if parameter == "values":
            data = block.values
        else:
            data = block.gradient(parameter).data
        M.append(data.reshape(np.prod(data.shape[:-1]), data.shape[-1]))
    return np.vstack(M)


def matrix_to_block(
    a: np.ndarray, sample_name: str = "sample", property_name: str = "property"
) -> TensorBlock:
    """Create a :class:`equistore.TensorBlock` from 2d :class`numpy.ndarray`.

    The values of the block are the same as `a`. The name of the property labels
    is `'property' and name of the sample labels are `'sample'`. The block has
    no components.

    :param a:
        2d numpy array for Blocks values
    :param sample_name:
        name of the TensorBlocks' samples
    :param property_name:
        name of the TensorMaps' properties

    :returns block:
        block with filled values

    Example:
    >>> a = np.zeros([2,2])
    >>> block = matrix_to_block(a)
    >>> print(block)
    """

    if len(a.shape) != 2:
        raise ValueError(f"`a` has {len(a.shape)} but must have exactly 2")

    n_samples, n_properties = a.shape

    samples = Labels([sample_name], np.arange(n_samples).reshape(-1, 1))
    properties = Labels([property_name], np.arange(n_properties).reshape(-1, 1))

    block = TensorBlock(
        values=a,
        samples=samples,
        components=[],
        properties=properties,
    )

    return block


def tensor_to_tensormap(a: np.ndarray, key_name: str = "keys") -> TensorMap:
    """Create a :class:`equistore.TensorMap` from 3d :class`numpy.ndarray`.

    First dimension of a defines the number of blocks.
    The values of each block are taken from the second and the third dimension.
    The name of the property labels in each block is `'property' and name of the sample
    labels is `'sample'`. The blocks have no components.

    :param a:
        3d numpy array for the block of the TensorMap values
    :param key_name:
        name of the TensorMaps' keys

    :returns:
        TensorMap with filled values


    Example:
    >>> a = np.zeros([2,2])
    >>> # make 2d array 3d tensor
    >>> tensor = tensor_to_tensormap(a[np.newaxis, :])
    >>> print(tensor)
    """
    if len(a.shape) != 3:
        raise ValueError(f"`a` has {len(a.shape)} but must have exactly 3")

    blocks = []
    for values in a:
        blocks.append(matrix_to_block(values))

    keys = Labels([key_name], np.arange(len(blocks)).reshape(-1, 1))
    return TensorMap(keys, blocks)


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
    save(tmp_filename, tensor_map)
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
        consistent with equistore.io.save format

    :returns tensor_map_dict:
        :class:`equistore.TensorMap` for the transform
    """
    tmp_filename = tempfile.mktemp() + ".npz"
    np.savez(tmp_filename, **tensor_map_dict)
    return load(tmp_filename)
