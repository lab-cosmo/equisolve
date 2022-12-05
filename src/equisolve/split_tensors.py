from typing import List
import warnings

import numpy as np

from equistore import TensorBlock, TensorMap
from equistore.operations import split


def split_tensors(
    tensors: List[TensorMap],
    axis: str = "samples",
    names: List[str] = ["structure"],
    n_groups: int = None,
    group_sizes_abs: List[int] = None,
    group_sizes_rel: List[float] = None,
    random_seed: int = None,
):
    """
    Take an input :py:class:`TensorMap` or list of :py:class:`TensorMap` objects
    and splits each of them into multiple :py:class:`TensorMap` objects based on
    some splitting criteria.

    If passing a list of :py:class:`TensorMap` objects, every
    :py:class:`TensorMap` must have the same set of unique indices for the
    specified ``axis`` and ``names``. They will all then be split in the same
    way.

    For instance, this function can be used to split 2 tensors corresponding to
    X and Y data into train and test data, as follows:

    .. code-block:: python

        [[X_train, X_test], [Y_train, Y_test]] = split(
            tensors=[X, Y], axis="samples", names=["structure"], n_groups=2,  #
            for each tensor we want a train and test set group_sizes_rel=[0.8,
            0.2],  # 80:20 train:test split
        )

    The function finds the unique indices in the tensor(s) that correspond to
    the chosen axis and names. These, if specified, are randomly shuffled and
    split into groups that dictate the indices that will appear in each of the
    output tensors.

    This function works by calling the :py:func:`slice` function and upon
    slicing along the samples/properties axis may produce empty
    :py:class:`TensorBlocks` in the resulting :py:class:`TensorMap` objects.

    :param tensors: input ``list`` of :py:class:`TensorMap` objects, each of
        which will be split into ``n_groups`` new :py:class:`TensorMap` objects.
    :param axis: a str equal to either 'samples' or 'properties'. This is the
        axis along which the input :py:class:`TensorMap` objects will be split.
    :param names: a ``list`` of ``str`` indicating the samples/properties names
        by which the tensors will be split.
    :param n_groups: an ``int`` indicating how many new :py:class:`TensorMap`
        objects each of the tensors passed in ``tensors`` will be split into.
    :param group_sizes_abs:
    :param group_sizes_rel:
    :param random_seed: an int that seeds the numpy random number generator.
        Used to control shuffling of the unique indices, which dictate the data
        that ends up in each of the split output tensors. If None, no shuffling
        of the indices occurs. If -1, shuffling occurs but with no random seed
        set. If an integer != -1, shuffling is executed but with a random seed
        set to this value.

    :return: list of list of :py:class:`TensorMap`. The ith element in the list
        contains ``n_groups`` :py:class:`TensorMap` objects corresponding to the
        split ith :py:class:`TensorMap` of the input list ``tensors``.
    """

    if isinstance(tensors, TensorMap):
        tensors = [tensors]

    # Perform input checks
    _check_split_args(
        tensors,
        axis,
        names,
        n_groups,
        group_sizes_abs,
        group_sizes_rel,
        random_seed,
    )

    # Get array of unique indices to split by
    unique_idxs = get_unique_indices(tensors, axis, names)
    n_indices = len(unique_idxs)

    if n_indices < n_groups:
        raise ValueError(
            "you must specify at least as many groups as there are indices to split."
        )

    if group_sizes_abs is None:
        if group_sizes_rel is None:
            # Equally sized groups
            group_sizes = np.array([1 / n_groups] * n_groups) * n_indices
            # Raise UserWarning if groups won't be exactly equally sized, i.e.
            # if n_indices isn't a multiple of n_groups
            if n_indices % n_groups != 0:
                warnings.warn(
                    f"you specified {n_groups} groups to split each tensor into. "
                    + "The resulting groups will not be exactly equally sized as "
                    + f"there are {n_indices} indices to split by."
                )
        else:
            # Convert relative group sizes to absolute group sizes
            group_sizes = np.array(group_sizes_rel) * n_indices
    else:
        group_sizes = np.array(group_sizes_abs)

    # The group sizes may not be integers. Use cascade rounding to round them
    # all to integers whilst attempting to minimize rounding error.
    group_sizes = _cascade_round(group_sizes)

    # Randomly shuffle the unique indices and group them accordingly
    grouped_idxs = _group_indices(unique_idxs, group_sizes, random_seed)

    # Split each of the input TensorMaps
    split_tensors = []
    for i, tensor in enumerate(tensors):
        with warnings.catch_warnings(record=True) as w:
            split_tensors.append(split(tensor, axis, names, grouped_idxs))

        if len(w) > 0:
            warnings.warn(
                f"upon splitting the TensorMap at index {i} in your input list "
                + "`tensors`, some of the blocks in the output (i.e. split) "
                + "TensorMaps are now empty."
            )

    return split_tensors


def get_unique_indices(
    tensors: List[TensorMap],
    axis: str,
    names: List[str],
) -> np.ndarray:
    """
    For a given :py:class:`TensorMap` or list of :py:class:`TensorMap` objects,
    finds all the unique samples/properties indices for the names passed. For
    instance, using ``axis='samples'`` and ``names='structure'``, then the
    unique structure indices present across all blocks of the
    :py:class:`TensorMap` are returned. If passed as a list, i.e.
    ``names=['structure', 'center']``, then all unique combinations of these
    samples names will be returned. This also works for properties, passing
    ``axis='properties'`` and then a str or list of properties names.

    If `tensors` is passed as a list of :py:class:`TensorMap` objects, this
    function will check that the unique indices calculated are exactly
    equivalent for all tensors in the list.

    :param tensormap : input :py:class:`TensorMap` or list of
        :py:class:`TensorMap` objects.
    :param axis: a `str` indicating which axis to find the
    :param names: str or list of str The samples/properties names(s) to find
        unique values for across every block of the TensorMap

    :return: a :py:class:`np.ndarray` 2D-array of unique indices for the
        samples/properties names passed as input.
    """

    if isinstance(tensors, TensorMap):
        tensors = [tensors]

    _check_get_unique_indices_args(tensors, axis, names)

    # Calculate unique indices for each TensorMap in the list
    list_unique_idxs = []
    for tensor in tensors:
        indices = set()
        for _, block in tensor:
            vals = (
                block.samples[names] if axis == "samples" else block.properties[names]
            )
            indices.update([i for i in vals])
        list_unique_idxs.append(list(indices))

    # Now check the indices for every TensorMap passed are exactly equivalent
    unique_idxs = list_unique_idxs[0]  # use the first set of indices as the reference
    if len(tensors) > 1:
        for i in range(1, len(tensors)):
            if not np.array_equal(unique_idxs, list_unique_idxs[i]):
                raise ValueError(
                    f"For your chosen input axis {axis} and names {names}, "
                    + "the input TensorMaps do not have the same combination of "
                    + "unique indices to slice by. Please check your input "
                    + "tensors and try again."
                )
    # Convert indices to a 2D numpy array. Makes it easier to instantiate a
    # Labels object. Sort values in ascending order.
    unique_idxs = np.sort(
        np.array([np.array([j for j in i]) for i in list(unique_idxs)]), axis=0
    )

    return unique_idxs


def _check_split_args(
    tensors: List[TensorMap],
    axis: str,
    names: List[str],
    n_groups: int,
    group_sizes_abs: List[int],
    group_sizes_rel: List[float],
    random_seed: int,
):

    # Check tensors passed as a list
    if not isinstance(tensors, list):
        raise TypeError("you must pass `tensors` as a list.")
    # Check all tensors in the list are TensorMaps
    for tensor in tensors:
        if not isinstance(tensor, TensorMap):
            raise TypeError(
                "you must pass `tensors` as a list of equistore `TensorMap` objects."
            )
    # Check axis
    if axis not in ["samples", "properties"]:
        raise ValueError("`axis` must be passsed as either 'samples' or 'properties'.")
    # Check names
    if not isinstance(names, list):
        raise TypeError("you must pass `names` as a list.")
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
                        + "`names`."
                    )
    # Check n_groups
    if not isinstance(n_groups, int):
        raise TypeError("`n_groups` must be passed as an int.")

    # Check group_sizes_abs and group_sizes_rel
    if group_sizes_abs is not None:
        # Length is the same as n_groups
        if len(group_sizes_abs) != n_groups:
            raise ValueError(
                "if specifying `group_sizes_abs` to split the tensors by, the list must "
                + "have the same number of elements as indicated by `n_groups`."
            )
        # group_sizes_rel isn't specified too
        if group_sizes_rel is not None:
            raise ValueError(
                "can only specify `group_sizes_abs` or `group_sizes_rel`, "
                + "but not both."
            )
    else:
        if group_sizes_rel is not None:
            # Length is the same as n_groups
            if len(group_sizes_rel) != n_groups:
                raise ValueError(
                    "if specifying `group_sizes_rel` to split the tensors by, the list "
                    + "must have the same number of elements as indicated by "
                    + "`n_groups`."
                )
            # Sum of the parts is 1
            if np.sum(group_sizes_rel) != 1:
                raise ValueError(
                    "if specifying `group_sizes_rel`, the values in the list must "
                    + "add to 1."
                )
    # Check random_seed
    if random_seed is not None:
        if not isinstance(random_seed, int):
            raise TypeError("`random_seed` must be passed as an `int`.")


def _check_get_unique_indices_args(
    tensors: list,
    axis: str,
    names: List[str],
):

    # Check tensors
    if not isinstance(tensors, list):
        raise TypeError("you must pass the `tensors` as a list of TensorMap objects.")
    # Check axis
    if not isinstance(axis, str):
        raise TypeError("you must pass `axis` as a str.")
    if axis not in ["samples", "properties"]:
        raise ValueError("`axis` must be passsed as either 'samples' or 'properties'.")
    # Check names
    if not isinstance(names, list):
        raise TypeError("you must pass `names` as a list of str.")
    for tensor in tensors:
        for _, block in tensor:
            tmp_names = (
                block.samples.names if axis == "samples" else block.properties.names
            )
            for name in names:
                if name not in tmp_names:
                    raise ValueError(
                        "each block of each `TensorMap` passed must have a samples "
                        + "or properties name that matches the one passed in `names`."
                    )


def _group_indices(indices, group_sizes: List[int], random_seed: int = None):
    """
    Takes an unshuffled list of indices, shuffles them, and splits them into
    n_groups number of groups, where n_groups is the length of `group_sizes`.
    The sum of the elements of `group_sizes` should equal the number of indices,
    n_indices.

    :param indices:
    :param group_sizes:
    :param random_seed: an ``int``. If None, no shuffling of the indices occurs.
        If -1, shuffling occurs but with no random seed set. If an integer !=
        -1, shuffling is executed but with a random seed set to this value.

    :return: a :py:class:`np.ndarray`
    """

    n_indices = len(indices)

    if np.sum(group_sizes) != n_indices:
        raise ValueError(
            "the sum of the absolute group sizes must add up to the "
            + "total number of indices being split."
        )

    # Randomly shuffle the unique indices
    if random_seed is not None:  # shuffle
        if random_seed != -1:  # set a numpy random seed
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Group the indices
    grouped_idxs = []
    prev_size = 0
    for i, size in enumerate(group_sizes):
        grouped_idxs.append(indices[prev_size : prev_size + size])
        prev_size += size

    return grouped_idxs


def _cascade_round(array: np.array):
    """
    Given an array of floats that sum to an integer, this rounds the floats
    and returns an array of integers with the same sum.

    Adapted from https://jsfiddle.net/cd8xqy6e/.
    """

    if not isinstance(array, np.ndarray):
        raise TypeError("must pass `array` as a numpy array.")
    mod = np.sum(array) % 1
    if not np.isclose(round(mod) - mod, 0):
        raise ValueError("elements of `array` must sum to an integer.")

    float_total = 0
    integer_total = 0
    rounded_array = []

    for element in array:
        new_integer = round(element + float_total) - integer_total
        float_total += element
        integer_total += new_integer
        rounded_array.append(new_integer)

    # Check that the sum is preserved
    assert round(np.sum(array)) == round(np.sum(rounded_array))

    return np.array(rounded_array)
