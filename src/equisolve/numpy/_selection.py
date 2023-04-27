# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
from typing import Type

import numpy as np
import skmatter._selection
from equistore import Labels, TensorBlock, TensorMap
from equistore.operations import slice_block


class GreedySelector:
    """Wraps :py:class:`skmatter._selection.GreedySelector` for a TensorMap.

    The class creates a selector for each block. The selection will be done based
    the values of each :py:class:`TensorBlock`. Gradients will not be considered for
    the selection.
    """

    def __init__(
        self,
        selector_class: Type[skmatter._selection.GreedySelector],
        selection_type: str,
        **selector_arguments,
    ) -> None:
        self._selector_class = selector_class
        self._selection_type = selection_type
        self._selector_arguments = selector_arguments

        self._selector_arguments["selection_type"] = self._selection_type
        self._support = None

    @property
    def selector_class(self) -> Type[skmatter._selection.GreedySelector]:
        """The class to perform the selection."""
        return self._selector_class

    @property
    def selection_type(self) -> str:
        """Whether to choose a subset of columns ('feature') or rows ('sample')."""
        return self._selection_type

    @property
    def selector_arguments(self) -> dict:
        """Arguments passed to the ``selector_class``."""
        return self._selector_arguments

    @property
    def support(self) -> TensorMap:
        """TensorMap containing the support."""
        if self._support is None:
            raise ValueError("No selections. Call fit method first.")

        return self._support

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

        blocks = []
        for _, block in X:
            selector = self.selector_class(**self.selector_arguments)
            selector.fit(block.values, warm_start=warm_start)
            mask = selector.get_support()

            if self._selection_type == "feature":
                samples = Labels.single()
                properties = block.properties[mask]
            elif self._selection_type == "sample":
                samples = block.samples[mask]
                properties = Labels.single()

            blocks.append(
                TensorBlock(
                    values=np.zeros([len(samples), len(properties)], dtype=np.int32),
                    samples=samples,
                    components=[],
                    properties=properties,
                )
            )

        self._support = TensorMap(X.keys, blocks)

        return self

    def transform(self, X: TensorMap) -> TensorMap:
        """Reduce X to the selected features.

        :param X:
            The input tensor.
        :returns:
            The selected subset of the input.
        """
        blocks = []
        for key, block in X:
            block_support = self.support.block(key)

            if self._selection_type == "feature":
                blocks.append(slice_block(block, properties=block_support.properties))
            elif self._selection_type == "sample":
                blocks.append(slice_block(block, samples=block_support.samples))

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
