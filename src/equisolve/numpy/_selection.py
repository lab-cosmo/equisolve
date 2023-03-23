# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from equistore import TensorMap, slice_block
from skmatter._selection import GreedySelector


class GreedySelector:
    """Wrapper for :py:class:`skmatter._selection.GreedySelector` for a
    :py:class:equistore.TensorMap`.

    Create a selector for each block. The selection will be done based
    the values of each :py:class:`TensorBlock`.
    """

    def __init__(
        self, selector: GreedySelector, selection_type: str, **selector_arguments
    ) -> None:
        self.selection_type = selection_type
        self.selector_arguments = selector_arguments
        self.selector = selector

        self.selector_arguments["selection_type"] = self.selection_type
        self.selectors = {}

    def fit(self, X: TensorMap, warm_start: bool = False) -> None:
        """Learn the features to select.

        :param X:
            Training vectors.
        :param warm_start:
            Whether the fit should continue after having already run, after increasing
            `n_to_select`. Assumes it is called with the same X.
        """
        if len(X.components_names) != 0:
            raise ValueError("Only blocks with no components are supported.")

        for key, block in X:
            selector = self.selector(**self.selector_arguments)
            self.selectors[key] = selector.fit(block.values, warm_start=warm_start)

        return self

    def transform(self, X: TensorMap) -> TensorMap:
        """Reduce X to the selected features.

        :param X:
            The input tensor.
        :returns:
            The selected subset of the input.
        """
        if len(self.selectors) == 0:
            raise ValueError("No selections. Call fit method first.")

        blocks = []
        for key, block in X:
            try:
                selector = self.selectors[key]
            except KeyError as err:
                raise ValueError(f"Block with key {key} does not exist.") from err

            mask = selector.get_support()

            if self.selection_type == "feature":
                blocks.append(slice_block(block, properties=block.properties[mask]))
            elif self.selection_type == "sample":
                blocks.append(slice_block(block, samples=block.samples[mask]))

        return TensorMap(X.keys, blocks)

    def fit_transform(self, X: TensorMap, warm_start: bool = False) -> TensorMap:
        """Fit to data, then transform it.

        :param X:
            Training vectors.
        :param warm_start:
            Whether the fit should continue after having already run, after increasing
            `n_to_select`. Assumes it is called with the same X.
        """
        return self.fit(X, warm_start=warm_start).transform(X)
