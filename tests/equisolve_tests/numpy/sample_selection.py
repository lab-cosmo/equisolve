# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
import metatensor
import numpy as np
import pytest
import skmatter.sample_selection
from metatensor import Labels
from numpy.testing import assert_equal, assert_raises

from equisolve.numpy.sample_selection import CUR, FPS

from .utilities import (
    random_single_block_no_components_tensor_map,
    random_tensor_map_with_components,
)


class TestSelection:
    @pytest.fixture
    def X1(self):
        return random_single_block_no_components_tensor_map()

    @pytest.fixture
    def X2(self):
        return random_tensor_map_with_components()

    @pytest.mark.parametrize(
        "selector_class, skmatter_selector_class",
        [(FPS, skmatter.sample_selection.FPS), (CUR, skmatter.sample_selection.CUR)],
    )
    def test_fit(self, X1, selector_class, skmatter_selector_class):
        selector = selector_class(n_to_select=2)
        selector.fit(X1)
        support = selector.support[0].samples

        skmatter_selector = skmatter_selector_class(n_to_select=2)
        skmatter_selector.fit(X1[0].values)
        skmatter_support = skmatter_selector.get_support(indices=True)
        skmatter_support_labels = Labels(
            names=["sample", "structure"],
            values=np.array(
                [[support_i, support_i] for support_i in skmatter_support],
                dtype=np.int32,
            ),
        )

        assert support == skmatter_support_labels

    @pytest.mark.parametrize(
        "selector_class, skmatter_selector_class",
        [(FPS, skmatter.sample_selection.FPS), (CUR, skmatter.sample_selection.CUR)],
    )
    def test_transform(self, X1, selector_class, skmatter_selector_class):
        selector = selector_class(n_to_select=2, random_state=0)
        selector.fit(X1)
        X_trans = selector.transform(X1)

        skmatter_selector = skmatter_selector_class(n_to_select=2, random_state=0)
        skmatter_selector.fit(X1[0].values)
        X_trans_skmatter = X1[0].values[skmatter_selector.get_support()]
        assert_equal(X_trans[0].values, X_trans_skmatter)

    @pytest.mark.parametrize("selector_class", [FPS, CUR])
    def test_fit_transform(self, X1, selector_class):
        selector = selector_class(n_to_select=2)

        X_ft = selector.fit(X1).transform(X1)
        metatensor.equal_raise(selector.fit_transform(X1), X_ft)

    @pytest.mark.parametrize("selector_class", [FPS])
    def test_get_select_distance(self, X2, selector_class):
        selector = selector_class(n_to_select=3)
        selector.fit(X2)
        select_distance = selector.get_select_distance

        assert select_distance is not None

        # Check distances sorted in descending order, with an inf as the first
        # entry
        for block in select_distance:
            assert block.values[0][0] == np.inf
            for i, val in enumerate(block.values[0][1:], start=1):
                assert val < block.values[0][i - 1]

    @pytest.mark.parametrize("selector_class", [FPS])
    def test_get_select_distance_n_to_select(self, X2, selector_class):
        # Case 1: select all features for every block (n_to_select = -1)
        selector = selector_class(n_to_select=-1)
        selector.fit(X2)
        select_distance = selector.get_select_distance
        for block in select_distance:
            assert len(block.samples) == 4

        # Case 2: select subset of features but same for each block
        n = 2
        selector = selector_class(n_to_select=n)
        selector.fit(X2)
        select_distance = selector.get_select_distance
        for block in select_distance:
            assert len(block.samples) == n

        # Case 3: select subset of features but different for each block
        keys = X2.keys
        n = {tuple(key): i for i, key in enumerate(keys, start=1)}
        selector = selector_class(n_to_select=n)
        selector.fit(X2)
        select_distance = selector.get_select_distance
        for i, key in enumerate(keys, start=1):
            assert len(select_distance[key].samples) == i

    @pytest.mark.parametrize("selector_class", [CUR])
    def test_get_select_distance_raises(self, X2, selector_class):
        selector = selector_class(n_to_select=3)
        selector.fit(X2)
        with assert_raises(ValueError):
            selector.get_select_distance
