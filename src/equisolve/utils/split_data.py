# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
"""
Module for splitting lists of :py:class:`TensorMap` objects into multiple
:py:class:`TensorMap` objects along a given axis.
"""
from typing import List, Union, Optional, Tuple
import numpy as np

import equistore
from equistore import Labels, TensorBlock, TensorMap


def split_data(
    tensors: Union[List[TensorMap], TensorMap],
    axis: str,
    names: Union[List[str], str],
    n_groups: int,
    group_sizes_abs: Optional[List[int]] = None,
    group_sizes_rel: Optional[List[float]] = None,
    random_seed: Optional[int] = None,
) -> Tuple[List[List[TensorMap]], List[Labels]]:
    """
    Splits a list of :py:class:`TensorMap` objects into multiple
    :py:class:`TensorMap` objects along a given axis.

    For either the "samples" or "properties" `axis`, the unique indices for the
    specified metadata `name` are found. If `random_seed` is set, the indices
    are shuffled. Then, they are divided into `n_groups`, where the sizes of the
    groups are specified by either the `group_sizes_abs` or `group_sizes_rel`
    arguments.

    The grouped indices are then used to split the list of input tensors. The
    split tensors, along with the grouped indices, are returned. The tensors are
    returned as a list of list of :py:class:`TensorMap` objects.

    Each list in the returned list of list corresponds to the split
    :py:class`TensorMap` at the same position in the input `tensors` list. Each
    nested list contains :py:class:`TensorMap` objects that share no common
    indices for the specified `axis` and `names`. However, the metadata on all
    other axes (including the keys) will be equivalent.

    The passed list of :py:class:`TensorMap` objects in `tensors` must have the
    same set of unique indices for the specified `axis` and `names`. For
    instance, if passing an input and output tensor for splitting (i.e. as used
    in supervised machine learning), the output tensor must have structure
    indices 0 -> 10 if the input tensor does.

    :param tensors: input `list` of :py:class:`TensorMap` objects, each of
        which will be split into `n_groups` new :py:class:`TensorMap` objects.
    :param axis: a :py:class:`str` equal to either "samples or "properties". This is the
        axis along which the input :py:class:`TensorMap` objects will be split.
    :param names: a `list` of :py:class:`str` indicating the samples/properties
        names by which the `tensors` will be split.
    :param n_groups: an :py:class:`int` indicating how many new
        :py:class:`TensorMap` objects each of the tensors passed in `tensors`
        will be split into. If both `group_sizes_abs` and `group_sizes_rel`
        are not specified, `n_groups` is used to split the data into ``n`` evenly
        sized groups, to the nearest integer.
    :param group_sizes_abs: an ordered `list` of `float` the absolute group
        sizes to split each input TensorMap into. The sum of this list must be
        <= the total number of unique indices present in the TensorMaps for the
        chosen `axis` and `names`.
    :param group_sizes_rel: an ordered `list` of `float` of the relative
        group sizes to split each input TensorMap into. The sum of these float
        values must <= 1.
    :param random_seed: an int that seeds the numpy random number generator.
        Used to control shuffling of the unique indices, which dictate the data
        that ends up in each of the split output tensors. If None, no shuffling
        of the indices occurs. If -1, shuffling occurs but with no random seed
        set. If an integer != -1, shuffling is executed but with a random seed
        set to this value.

    :return: list of list of :py:class:`TensorMap`. The ith element in the list
        contains `n_groups` :py:class:`TensorMap` objects corresponding to the
        split ith :py:class:`TensorMap` of the input list `tensors`.
    :return: list of :py:class:`Labels` corresponding to the unique indices
        according to the specified `axis` and `names` that are present in
        each of the returned groups of :py:class:`TensorMap`. The length of this
        list is `n_groups`.

    Examples
    --------

    Split a TensorMap `tensor` into 2 new TensorMaps along the "samples" axis
    for the "structure" metadata. Without specifying `group_sizes_abs` or
    `group_sizes_rel`, the data will be split equally by structure index.
    Without specifying `random_seed`, no shuffling of the structure indices will
    occur and they will be grouped in lexigraphical order. For instance, if the
    input tensor has structure indices 0 -> 9 (inclusive), the first new tensor
    will contain only structure indices 0 -> 4 (inc.) and the second will
    contain only 5 -> 9 (inc).

    .. code-block:: python

        from equisolve.utils import split_data

        [[new_tensor_1, new_tensor_2]], grouped_idxs = split_data(
            tensors=tensor,
            axis="samples",
            names=["structure"],
            n_groups=2,
        )

    Split 2 tensors corresponding to input and output data into train and test
    data, with a relative 80:20 ratio. If both input and output tensors contain
    structure indices 0 -> 9 (inclusive), the `in_train` and `out_train` tensors
    will contain structure indices 0 -> 7 (inc.) and the `in_test` and
    `out_test` tensors will contain structure indices 8 -> 9 (inc.).

    .. code-block:: python

        from equisolve.utils import split_data

        [[in_train, in_test], [out_train, out_test]], grouped_idxs = split_data(
            tensors=[input, output],
            axis="samples",
            names=["structure"],
            n_groups=2,                  # for train-test split
            group_sizes_rel=[0.8, 0.2],  # 80:20 train:test split
            random_seed=100,
        )

    Split 2 tensors corresponding to input and output data into train, test, and
    validation data. If input and output tensors have the same 100 structure
    indices, we can split such that the train, test, and val tensors have 70,
    20, and 10 structures in each, respectively. Specifying the `random_seed`
    will shuffle the structure indices before they are grouped.

    .. code-block:: python

        from equisolve.utils import split_data

        (
            [
                [in_train, in_test, in_val],
                [out_train, out_test, out_val]
            ]
        ), grouped_idxs = split_data(
            tensors=[input, output],
            axis="samples",
            names=["structure"],
            n_groups=3,                    # for train-test-validation
            group_sizes_abs=[70, 20, 10],  # 70, 25, 10 train, test, val
            random_seed=100,
        )
    """
    # Check input args
    tensors = [tensors] if isinstance(tensors, TensorMap) else tensors
    names = [names] if isinstance(names, str) else names
    _check_args(
        tensors, axis, names, n_groups, group_sizes_abs, group_sizes_rel, random_seed
    )

    # Get array of unique indices to split by for each tensor in `tensors`
    unique_idxs_list = [
        equistore.unique_metadata(tensor, axis, names) for tensor in tensors
    ]

    # Check that the unique indices are equivalent for all input tensors
    _check_labels_equivalent(unique_idxs_list)
    unique_idxs = unique_idxs_list[0]

    # Shuffle the unique indices according to the random seed
    _shuffle_indices(unique_idxs, random_seed)

    # Get group sizes
    group_sizes = _get_group_sizes(
        n_groups, len(unique_idxs), group_sizes_abs, group_sizes_rel
    )

    # Group the indices according to the group sizes
    grouped_idxs = _group_indices(unique_idxs, group_sizes)

    # Split each of the input TensorMaps
    split_tensors = []
    for i, tensor in enumerate(tensors):
        split_tensors.append(equistore.split(tensor, axis, grouped_idxs))

    # Check the indices of the split tensors. the jth TensorMap in every list of
    # split tensors should have equivalent unique indices.
    for j in range(len(split_tensors[0])):
        unq_idxs_list = [
            equistore.unique_metadata(split_tensors[i][j], axis, names)
            for i in range(len(split_tensors))
        ]
        _check_labels_equivalent(unq_idxs_list)

    return split_tensors, grouped_idxs


def _shuffle_indices(indices: Labels, random_seed: Optional[int] = None):
    """
    Shuffles the input Labels object `indices` according to the random seed.
    If `random_seed=None`, not shuffling is performed and the indices are just
    returned. If -1, shuffling occurs but with no random seed set. If an integer
    != -1, shuffling is executed but with a random seed set to this value.
    """
    if random_seed is not None:  # shuffle
        if random_seed != -1:  # set a numpy random seed
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    return


def _get_group_sizes(
    n_groups: int,
    n_indices: int,
    group_sizes_abs: Optional[List[float]] = None,
    group_sizes_rel: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Parses the group sizes args from :py:func:`split_data` and returns an array
    of group sizes in absolute numbers.
    :param n_groups: an int, the number of groups to split the data into
    :param n_indices: an int, the number of unique indices present in the data
        by which the data should be grouped.
    :param group_sizes_abs:
    """
    # Must be at least as many unique indices as groups
    if not (n_indices >= n_groups):
        raise ValueError(
            "you must specify at least as many groups as there are indices to split."
        )
    if group_sizes_abs is None:
        if group_sizes_rel is None:
            # Equally sized groups
            group_sizes = np.array([1 / n_groups] * n_groups) * n_indices
        else:
            # Convert relative group sizes to absolute group sizes
            group_sizes = np.array(group_sizes_rel) * n_indices
    else:
        # Already in absolute; just convert to an array
        group_sizes = np.array(group_sizes_abs)

    # The group sizes may not be integers. Use cascade rounding to round them
    # all to integers whilst attempting to minimize rounding error.
    group_sizes = _cascade_round(group_sizes)

    return group_sizes


def _cascade_round(array: np.ndarray) -> np.ndarray:
    """
    Given an array of floats that sum to an integer, this rounds the floats
    and returns an array of integers with the same sum.
    Adapted from https://jsfiddle.net/cd8xqy6e/.
    """
    # Check type
    if not isinstance(array, np.ndarray):
        raise TypeError("must pass `array` as a numpy array.")
    # Check sum
    mod = np.sum(array) % 1
    if not np.isclose(round(mod) - mod, 0):
        raise ValueError("elements of `array` must sum to an integer.")

    float_tot, integer_tot = 0, 0
    rounded_array = []
    for element in array:
        new_int = round(element + float_tot) - integer_tot
        float_tot += element
        integer_tot += new_int
        rounded_array.append(new_int)

    # Check that the sum is preserved
    assert round(np.sum(array)) == round(np.sum(rounded_array))

    return np.array(rounded_array)


def _group_indices(indices: Labels, group_sizes: List[int]) -> Labels:
    """
    Sets a numpy `random_seed` then randomly shuffles `indices`. Next, these
    indices are split into smaller groups according to the sizes specified in
    `group_sizes`, and returned as a list of :py:class:`Labels` objects.
    """
    # Group the indices
    grouped_idxs = []
    prev_size = 0
    for i, size in enumerate(group_sizes):
        grouped_idxs.append(indices[prev_size : prev_size + size])
        prev_size += size

    return grouped_idxs


def _check_labels_equivalent(
    labels_list: List[Labels],
):
    """
    Checks that all Labels objects in the input List `labels_list` are
    equivalent in names and values.
    """
    if len(labels_list) <= 1:
        return
    # Define reference Labels object as the first in the list
    try:
        ref_labels = _searchable_labels(labels_list[0])
    except AttributeError:
        # Labels obj is already searchable
        ref_labels = labels_list[0]
    for label_i, test_label in enumerate(labels_list, start=1):
        if not np.array_equal(ref_labels, test_label):
            raise ValueError(
                f"Labels objects in `labels_list` are not equivalent: {ref_labels} != {test_label}"
            )


def _searchable_labels(labels: Labels):
    """
    Returns the input Labels object but after being used to construct a
    TensorBlock, so that look-ups can be performed.
    """
    return TensorBlock(
        values=np.full((len(labels), 1), 0.0),
        samples=labels,
        components=[],
        properties=Labels(["p"], np.array([[0]], dtype=np.int32)),
    ).samples


def _labels_equal(a: Labels, b: Labels, exact_order: bool):
    """
    For 2 :py:class:`Labels` objects `a` and `b`, returns true if they are
    exactly equivalent in names, values, and elemental positions. Assumes that
    the Labels are already searchable, i.e. they belogn to a parent TensorBlock
    or TensorMap.
    """
    # They can only be equivalent if the same length
    if len(a) != len(b):
        return False
    if exact_order:
        return np.all(np.array(a == b))
    else:
        return np.all([a_i in b for a_i in a])


def _check_args(
    tensors: List[TensorMap],
    axis: str,
    names: List[str],
    n_groups: int,
    group_sizes_abs: List[int],
    group_sizes_rel: List[float],
    random_seed: int,
):
    """Checks the input args for :py:func:`split_data`."""
    # Check tensors passed as a list
    if not isinstance(tensors, list):
        raise TypeError(
            f"`tensors` must be a list of equistore `TensorMap`, got {type(tensors)}"
        )
    # Check all tensors in the list are TensorMaps
    for tensor in tensors:
        if not isinstance(tensor, TensorMap):
            raise TypeError(
                f"`tensors` must bes a list of equistore `TensorMap`, got {type(tensors)}"
            )
    # Check axis
    if axis not in ["samples", "properties"]:
        raise ValueError(
            f"`axis` must be passsed as either 'samples' or 'properties', got {axis}"
        )
    # Check names
    if not isinstance(names, list):
        raise TypeError(f"`names` must be a list of str, got {type(names)}")
    for tensor in tensors:
        for _, block in tensor:
            tmp_names = (
                block.samples.names if axis == "samples" else block.properties.names
            )
            for name in names:
                if name not in tmp_names:
                    raise ValueError(
                        "each block of each `TensorMap` passed must have a samples "
                        + "or properties name that matches the one passed in "
                        + "`names`"
                    )
    # Check n_groups
    if not isinstance(n_groups, int):
        raise TypeError(f"`n_groups` must be passed as an int, got {type(n_groups)}")

    # Check group_sizes_abs and group_sizes_rel
    if group_sizes_abs is not None:
        # Length is the same as n_groups
        if len(group_sizes_abs) != n_groups:
            raise ValueError(
                "if specifying `group_sizes_abs` to split the tensors by, the list must"
                + " have the same number of elements as indicated by `n_groups`"
            )
        # group_sizes_rel isn't specified too
        if group_sizes_rel is not None:
            raise ValueError(
                "can only specify `group_sizes_abs` or `group_sizes_rel`, but not both"
            )
    else:
        if group_sizes_rel is not None:
            # Length is the same as n_groups
            if len(group_sizes_rel) != n_groups:
                raise ValueError(
                    "if specifying `group_sizes_rel` to split the tensors by, the list "
                    + "must have the same number of elements as indicated by `n_groups`"
                )
            # Sum of the parts is 1
            if np.sum(group_sizes_rel) > 1:
                raise ValueError(
                    "if specifying `group_sizes_rel`, the sum of the values in the list"
                    + " must be <= 1"
                )
    # Check random_seed
    if random_seed is not None:
        if not isinstance(random_seed, int):
            raise TypeError("`random_seed` must be passed as an `int`")
