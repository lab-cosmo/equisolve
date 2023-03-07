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
from typing import List, Optional, Tuple, Union

import equistore
import numpy as np
from equistore import Labels, TensorMap


def split_data(
    tensors: Union[List[TensorMap], TensorMap],
    axis: str,
    names: Union[List[str], str],
    n_groups: int,
    group_sizes: Optional[Union[List[int], List[float]]] = None,
    random_seed: Optional[int] = None,
) -> Tuple[List[List[TensorMap]], List[Labels]]:
    """
    Splits a list of :py:class:`TensorMap` objects into multiple
    :py:class:`TensorMap` objects along a given axis.

    For either the "samples" or "properties" `axis`, the unique indices for the
    specified metadata `name` are found. If `random_seed` is set, the indices
    are shuffled. Then, they are divided into `n_groups`, where the sizes of the
    groups are specified by the `group_sizes` argument.

    These grouped indices are then used to split the list of input tensors. The
    split tensors, along with the grouped labels, are returned. The tensors are
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

    :param tensors: input `list` of :py:class:`TensorMap` objects, each of which
        will be split into `n_groups` new :py:class:`TensorMap` objects.
    :param axis: a :py:class:`str` equal to either "samples or "properties".
        This is the axis along which the input :py:class:`TensorMap` objects
        will be split.
    :param names: a `list` of :py:class:`str` indicating the samples/properties
        names by which the `tensors` will be split.
    :param n_groups: an :py:class:`int` indicating how many new
        :py:class:`TensorMap` objects each of the tensors passed in `tensors`
        will be split into. If `group_sizes` is none (default), `n_groups` is
        used to split the data into ``n`` evenly sized groups according to the
        unique metadata for the specified `axis` and `names`, to the nearest
        integer.
    :param group_sizes: an ordered `list` of `float` the group sizes to split
        each input TensorMap into. A list of int will be interpreted as an
        indication of the absolute group sizes, whereas a list of float as
        indicating the relative sizes. For the former case, the sum of this list
        must be <= the total number of unique indices present in the input
        `tensors` for the chosen `axis` and `names`. In the latter, the sum of
        this list must be <= 1.
    :param random_seed: an int that seeds the numpy random number generator.
        Used to control shuffling of the unique indices, which dictate the data
        that ends up in each of the split output tensors. If None, no shuffling
        of the indices occurs. If -1, shuffling occurs but with no random seed
        set. If an integer != -1, shuffling is executed but with a random seed
        set to this value.

    :return split_tensors: list of list of :py:class:`TensorMap`. The ith
        element in the list contains `n_groups` :py:class:`TensorMap` objects
        corresponding to the split ith :py:class:`TensorMap` of the input list
        `tensors`.
    :return grouped_labels: list of :py:class:`Labels` corresponding to the
        unique indices according to the specified `axis` and `names` that are
        present in each of the returned groups of :py:class:`TensorMap`. The
        length of this list is `n_groups`.

    Examples
    --------

    Split a TensorMap `tensor` into 2 new TensorMaps along the "samples" axis
    for the "structure" metadata. Without specifying `group_sizes`, the data
    will be split equally by structure index. If the number of unique strutcure
    indices present in the input data is not exactly divisible by `n_groups`,
    the group sizes will be made to the nearest int. Without specifying
    `random_seed`, no shuffling of the structure indices will occur and they
    will be grouped in lexigraphical order. For instance, if the input tensor
    has structure indices 0 -> 9 (inclusive), the first new tensor will contain
    only structure indices 0 -> 4 (inc.) and the second will contain only 5 -> 9
    (inc).

    .. code-block:: python

        from equisolve.utils import split_data

        [[new_tensor_1, new_tensor_2]], grouped_labels = split_data(
            tensors=tensor,
            axis="samples",
            names=["structure"],
            n_groups=2,
        )

    Split 2 tensors corresponding to input and output data into train and test
    data, with a relative 80:20 ratio. If both input and output tensors contain
    structure indices 0 -> 9 (inclusive), the `in_train` and `out_train` tensors
    will contain structure indices 0 -> 7 (inc.) and the `in_test` and
    `out_test` tensors will contain structure indices 8 -> 9 (inc.). As we want
    to specify relative group sizes, we will pass `group_sizes` as a list of
    float. Specifying the `random_seed` will shuffle the structure indices
    before the groups are made.

    .. code-block:: python

        from equisolve.utils import split_data

        [[in_train, in_test], [out_train, out_test]], grouped_labels = split_data(
            tensors=[input, output],
            axis="samples",
            names=["structure"],
            n_groups=2,                  # for train-test split
            group_sizes=[0.8, 0.2],  # relative, a 80% 20% train-test split
            random_seed=100,
        )

    Split 2 tensors corresponding to input and output data into train, test, and
    validation data. If input and output tensors have the same 10 structure
    indices, we can split such that the train, test, and val tensors have 7,
    2, and 1 structures in each, respectively. We want to specify absolute
    group sizes, so will pass a list of int. Specifying the `random_seed` will
    shuffle the structure indices before they are grouped.

    .. code-block:: python

        import equistore
        from equisolve.utils import split_data

        # Find the unique structure indices in the input tensor
        unique_structure_indices = equistore.unique_metadata(
            tensor=input, axis="samples", names=["structure"],
        )
        # They run from 0 -> 10 (inclusive)
        unique_structure_indices
        >>> Labels(
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (8,), (9,)],
            dtype=[('structure', '<i4')]
        )
        # Verify that the output has the same unique structure indices
        assert unique_structure_indices == equistore.unique_metadata(
            tensor=output, axis="samples", names=["structure"],
        )
        >>> True

        # Split the data by structure index, with an abolute split of 7, 2, 1
        # for the train, test, and validation tensors, respectively
        (
            [
                [in_train, in_test, in_val],
                [out_train, out_test, out_val]
            ]
        ), grouped_labels = split_data(
            tensors=[input, output],
            axis="samples",
            names=["structure"],
            n_groups=3,  # for train-test-validation
            group_sizes=[7, 2, 1],  # absolute; 7, 2, 1 for train, test, val
            random_seed=100,
        )
        # Inspect the grouped structure indices
        grouped_labels
        >>> [
            Labels(
                [(3,), (7,), (1,), (8,), (0,), (9,), (2,)],
                dtype=[('structure', '<i4')]
            ),
            Labels([(4,), (6,)], dtype=[('structure', '<i4')]),
            Labels([(5,)], dtype=[('structure', '<i4')]),
        ]
    """
    # Check input args and parse `tensors` and `names` into lists
    tensors = [tensors] if isinstance(tensors, TensorMap) else tensors
    names = [names] if isinstance(names, str) else names
    _check_args(tensors, axis, names, n_groups, group_sizes, random_seed)

    # Get array of unique indices to split by for each tensor in `tensors`
    unique_idxs_list = [
        equistore.unique_metadata(tensor, axis, names) for tensor in tensors
    ]

    # Check that the unique indices are equivalent for all input tensors
    _check_labels_equivalent(unique_idxs_list)
    unique_idxs = unique_idxs_list[0]

    # Shuffle the unique indices according to the random seed
    _shuffle_indices(unique_idxs, random_seed)

    # Must be at least as many unique indices as groups
    n_indices = len(unique_idxs)
    if n_indices < n_groups:
        raise ValueError(
            f"the number of groups specified ({n_groups}) is greater than the"
            + f" number of unique metadata indices ({n_indices}) for the"
            + f" chosen axis {axis} and names {names}: {unique_idxs}"
        )

    # Get group sizes
    group_sizes = _get_group_sizes(n_groups, len(unique_idxs), group_sizes)

    # The sum of the absolute group sizes must be less than or equal to the
    # number of unique indices
    if n_indices < sum(group_sizes):
        raise ValueError(
            f"the sum of the absolute group sizes ({sum(group_sizes)}) is greater than "
            + f"the number of unique metadata indices ({n_indices}) for the chosen "
            + f"axis {axis} and names {names}: {unique_idxs}"
        )

    # Group the indices according to the group sizes
    grouped_labels = _group_indices(unique_idxs, group_sizes)

    # Split each of the input TensorMaps
    split_tensors = []
    for i, tensor in enumerate(tensors):
        split_tensors.append(equistore.split(tensor, axis, grouped_labels))

    # Check the indices of the split tensors. the jth TensorMap in every list of
    # split tensors should have equivalent unique indices.
    for j in range(len(split_tensors[0])):
        unq_idxs_list = [
            equistore.unique_metadata(split_tensors[i][j], axis, names)
            for i in range(len(split_tensors))
        ]
        _check_labels_equivalent(unq_idxs_list)

    return split_tensors, grouped_labels


def _shuffle_indices(indices: Labels, random_seed: Optional[int] = None):
    """
    Shuffles the input Labels object `indices` according to the random seed. If
    `random_seed=None`, not shuffling is performed and the indices are just
    returned. If -1, shuffling occurs but with no random seed set. If an integer
    != -1, shuffling is executed but with a random seed set to this value. There
    is no return value, as the input `indices` is shuffled in place.
    """
    if random_seed is not None:  # shuffle
        if random_seed != -1:  # set a numpy random seed
            np.random.seed(random_seed)
        np.random.shuffle(indices)


def _get_group_sizes(
    n_groups: int,
    n_indices: int,
    group_sizes: Optional[Union[List[float], List[int]]] = None,
) -> np.ndarray:
    """
    Parses the `group_sizes` arg from :py:func:`split_data` and returns an array
    of group sizes in absolute numbers.

    :param n_groups: an int, the number of groups to split the data into :param
        n_indices: an int, the number of unique indices present in the data by
        which the data should be grouped.
    :param n_indices: a :py:class:`int` for the number of unique indices present
        in the input data for the specified `axis` and `names`.
    :param group_sizes: a :py:class:`list` of :py:class:`float` or
        :py:class:`int` indicating the absolute or relative group sizes,
        respectively.

    :return: a :py:class:`numpy.ndarray` of :py:class:`int` indicating the
        absolute group sizes.
    """
    if group_sizes is None:  # equally sized groups
        group_sizes = np.array([1 / n_groups] * n_groups) * n_indices
    elif np.all([isinstance(size, int) for size in group_sizes]):  # absolute
        group_sizes = np.array(group_sizes)
    else:  # relative; list of float
        group_sizes = np.array(group_sizes) * n_indices

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
    grouped_labels = []
    prev_size = 0
    for size in group_sizes:
        grouped_labels.append(indices[prev_size : prev_size + size])
        prev_size += size

    return grouped_labels


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
    ref_labels = labels_list[0]
    for label_i in range(1, len(labels_list)):
        test_label = labels_list[label_i]
        if not np.array_equal(ref_labels, test_label):
            raise ValueError(
                "Labels objects in `labels_list` are not equivalent:"
                + f" {ref_labels} != {test_label}"
            )


def _check_args(
    tensors: List[TensorMap],
    axis: str,
    names: List[str],
    n_groups: int,
    group_sizes: Optional[Union[List[float], List[int]]],
    random_seed: Optional[int],
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
                "`tensors` must be a list of equistore `TensorMap`,"
                + f" got {type(tensors)}"
            )
    # Check axis
    if not isinstance(axis, str):
        raise TypeError(f"`axis` must be passed as a str, got {type(axis)}")
    if axis not in ["samples", "properties"]:
        raise ValueError(
            f"`axis` must be passsed as either 'samples' or 'properties', got {axis}"
        )
    # Check names
    if not isinstance(names, list):
        raise TypeError(f"`names` must be a list of str, got {type(names)}")
    if not all([isinstance(name, str) for name in names]):
        raise TypeError(f"`names` must be a list of str, got {type(names)}")
    for tensor in tensors:
        tmp_names = tensor.sample_names if axis == "samples" else tensor.property_names
        for name in names:
            if name not in tmp_names:
                raise ValueError(
                    f"the passed `TensorMap` objects have {axis} names {tmp_names}"
                    + f" that do not match the one passed in `names` {names}"
                )
    # Check n_groups
    if not isinstance(n_groups, int):
        raise TypeError(f"`n_groups` must be passed as an int, got {type(n_groups)}")
    if not n_groups > 0:
        raise ValueError(f"`n_groups` must be greater than 0, got {n_groups}")
    # Check group_sizes
    if group_sizes is not None:
        if not isinstance(group_sizes, list):
            raise TypeError(
                "`group_sizes` must be passed as a list of float or int,"
                + f" got {type(group_sizes)}"
            )
        if len(group_sizes) != n_groups:
            raise ValueError(
                "if specifying `group_sizes`, you must pass a list whose"
                + " number of elements equal to `n_groups`"
            )
        for size in group_sizes:
            if not isinstance(size, (int, float)):
                raise TypeError(
                    "`group_sizes` must be passed as a list of float or int,"
                    + f" got {type(group_sizes)}"
                )
            if not size > 0:
                raise ValueError(
                    "all elements of `group_sizes` must be greater than 0,"
                    + f" got {group_sizes}"
                )
        if np.all([isinstance(size, float) for size in group_sizes]):
            if np.sum(group_sizes) > 1:
                raise ValueError(
                    "if specifying `group_sizes` as a list of float, the sum of"
                    + " the list must be less than or equal to 1"
                )
    # Check random_seed
    if random_seed is not None:
        if not isinstance(random_seed, int):
            raise TypeError(
                f"`random_seed` must be passed as an `int`, got {type(random_seed)}"
            )
