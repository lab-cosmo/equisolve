# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause


import equistore
import pytest
import skmatter.feature_selection
from numpy.testing import assert_equal

from equisolve.numpy.feature_selection import CUR, FPS

from .utils import random_single_block_no_components_tensor_map


class TestSelection:
    @pytest.fixture
    def X(self):
        return random_single_block_no_components_tensor_map()

    @pytest.mark.parametrize(
        "selector_cls, skmatter_cls",
        [(FPS, skmatter.feature_selection.FPS), (CUR, skmatter.feature_selection.CUR)],
    )
    def test_fit(self, X, selector_cls, skmatter_cls):
        selector = selector_cls(n_to_select=2)
        selector.fit(X)
        support = selector.selectors[X.keys[0]].get_support()

        skmatter_selector = skmatter_cls(n_to_select=2)
        skmatter_selector.fit(X[0].values)
        skmatter_support = skmatter_selector.get_support()

        assert_equal(support, skmatter_support)

    @pytest.mark.parametrize(
        "selector_cls, skmatter_cls",
        [(FPS, skmatter.feature_selection.FPS), (CUR, skmatter.feature_selection.CUR)],
    )
    def test_transform(self, X, selector_cls, skmatter_cls):
        selector = selector_cls(n_to_select=2)
        selector.fit(X)
        X_trans = selector.transform(X)

        skmatter_selector = skmatter_cls(n_to_select=2)
        skmatter_selector.fit(X[0].values)
        X_trans_skmatter = skmatter_selector.transform(X[0].values)

        assert_equal(X_trans[0].values, X_trans_skmatter)

    @pytest.mark.parametrize("selector_cls", [FPS, CUR])
    def test_fit_transform(self, X, selector_cls):
        selector = selector_cls(n_to_select=2)

        X_ft = selector.fit(X).transform(X)
        equistore.equal_raise(selector.fit_transform(X), X_ft)
