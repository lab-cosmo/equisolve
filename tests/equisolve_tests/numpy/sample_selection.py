# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
import equistore
import numpy as np
import pytest
import skmatter.sample_selection
from equistore import Labels
from numpy.testing import assert_equal

from equisolve.numpy.sample_selection import CUR, FPS

from .utilities import random_single_block_no_components_tensor_map


class TestSelection:
    @pytest.fixture
    def X(self):
        return random_single_block_no_components_tensor_map()

    @pytest.mark.parametrize(
        "selector_class, skmatter_selector_class",
        [(FPS, skmatter.sample_selection.FPS), (CUR, skmatter.sample_selection.CUR)],
    )
    def test_fit(self, X, selector_class, skmatter_selector_class):
        selector = selector_class(n_to_select=2)
        selector.fit(X)
        support = selector.support[0].samples["structure"]

        skmatter_selector = skmatter_selector_class(n_to_select=2)
        skmatter_selector.fit(X[0].values)
        skmatter_support = skmatter_selector.get_support(indices=True)
        skmatter_support_labels = Labels(
            names=["structure"],
            values=np.array(
                [[support_i] for support_i in skmatter_support], dtype=np.int32
            ),
        )

        assert_equal(support, skmatter_support_labels)

    @pytest.mark.parametrize(
        "selector_class, skmatter_selector_class",
        [(FPS, skmatter.sample_selection.FPS), (CUR, skmatter.sample_selection.CUR)],
    )
    def test_transform(self, X, selector_class, skmatter_selector_class):
        selector = selector_class(n_to_select=2)
        selector.fit(X)
        X_trans = selector.transform(X)

        skmatter_selector = skmatter_selector_class(n_to_select=2)
        skmatter_selector.fit(X[0].values)
        X_trans_skmatter = skmatter_selector.transform(X[0].values)

        assert_equal(X_trans[0].values, X_trans_skmatter)

    @pytest.mark.parametrize("selector_class", [FPS, CUR])
    def test_fit_transform(self, X, selector_class):
        selector = selector_class(n_to_select=2)

        X_ft = selector.fit(X).transform(X)
        equistore.equal_raise(selector.fit_transform(X), X_ft)
