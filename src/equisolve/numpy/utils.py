# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile

import metatensor
import numpy as np
from metatensor import Labels, TensorBlock, TensorMap


def array_from_block(block: TensorBlock) -> np.ndarray:
    """Extract parts of a :class:`metatensor.TensorBlock` into a array.

    All components will be stacked along the rows / "sample"-dimension.
    This means that the number of coulums / "property"-dimension is unchanged.

    :param block:
        :class:`metatensor.TensorBlock` for the extraction
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
    """Format an object of a :class:`metatensor.TensorBlock` into a dict of array.

    TODO rm usage of this function as soon
         https://github.com/lab-cosmo/metatensor/issues/94
         is merged

    :param tensor_map:
        :class:`metatensor.TensorMap` for the transform

    :returns tensor_map_dict:
        :class:`dict` of :class:`numpy.ndarray`, consistent with metatensor.io.save
        format
    """
    tmp_filename = tempfile.mktemp() + ".npz"
    metatensor.save(tmp_filename, tensor_map)
    tensor_map_dict = {key: value for key, value in np.load(tmp_filename).items()}
    os.remove(tmp_filename)
    return tensor_map_dict


def dict_to_tensor_map(tensor_map_dict: dict):
    """Format a dict of arrays complying with :class:`metatensor.TensorBlock`

    TODO rm usage of this function as soon
         https://github.com/lab-cosmo/metatensor/issues/94
         is merged

    :param tensor_map:
        :class:`dict` of :class:`numpy.ndarray`,
        consistent with :func:`metatensor.save` format

    :returns tensor_map_dict:
        :class:`metatensor.TensorMap` for the transform
    """
    tmp_filename = tempfile.mktemp() + ".npz"
    np.savez(tmp_filename, **tensor_map_dict)
    return metatensor.load(tmp_filename)


def core_tensor_map_to_torch(core_tensor: TensorMap, device=None, dtype=None):
    """Transforms a tensor map from metatensor-core to metatensor-torch

    :param core_tensor:
        tensor map from metatensor-core

    :param device:
        :py:class:`torch.device` of values in the resulting tensor map

    :param dtye:
        :py:class:`torch.dtype` of values in the resulting tensor map

    :returns torch_tensor:
        tensor map from metatensor-torch
    """
    from metatensor.torch import TensorMap as TorchTensorMap

    torch_blocks = []
    for _, core_block in core_tensor.items():
        torch_blocks.append(core_tensor_block_to_torch(core_block, device, dtype))
    torch_keys = core_labels_to_torch(core_tensor.keys)
    return TorchTensorMap(torch_keys, torch_blocks)


def core_tensor_block_to_torch(core_block: TensorBlock, device=None, dtype=None):
    """Transforms a tensor block from metatensor-core to metatensor-torch

    :param core_block:
        tensor block from metatensor-core

    :param device:
        :py:class:`torch.device` of values in the resulting block and labels

    :param dtye:
        :py:class:`torch.dtype` of values in the resulting block and labels

    :returns torch_block:
        tensor block from metatensor-torch
    """
    import torch
    from metatensor.torch import TensorBlock as TorchTensorBlock

    return TorchTensorBlock(
        values=torch.tensor(core_block.values, device=device, dtype=dtype),
        samples=core_labels_to_torch(core_block.samples, device=device),
        components=[
            core_labels_to_torch(component, device=device)
            for component in core_block.components
        ],
        properties=core_labels_to_torch(core_block.properties, device=device),
    )


def core_labels_to_torch(core_labels: Labels, device=None):
    """Transforms labels from metatensor-core to metatensor-torch

    :param core_block:
        tensor block from metatensor-core

    :param device:
        :py:class:`torch.device` of values in the resulting labels

    :returns torch_block:
        labels from metatensor-torch
    """
    import torch
    from metatensor.torch import Labels as TorchLabels

    return TorchLabels(
        core_labels.names, torch.tensor(core_labels.values, device=device)
    )


def transpose_tensor_map(tensor: TensorMap):
    blocks = []
    for block in tensor.blocks():
        block = TensorBlock(
            values=block.values.T,
            samples=block.properties,
            components=block.components,
            properties=block.samples,
        )
        blocks.append(block)
    return TensorMap(tensor.keys, blocks)
