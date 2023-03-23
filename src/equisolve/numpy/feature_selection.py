# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
"""Wrappers for the feature selectors of `scikit-matter`_.

.. _`scikit-matter`: https://scikit-matter.readthedocs.io/en/latest/selection.html
"""

from skmatter._selection import _CUR, _FPS

from ._selection import GreedySelector


class FPS(GreedySelector):
    """
    Transformer that performs Greedy Feature Selection using Farthest Point Sampling.

    Refer to :py:class:`skmatter.feature_selection.FPS` for full documentation.
    """

    def __init__(
        self,
        initialize=0,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selector_class=_FPS,
            selection_type="feature",
            initialize=initialize,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )


class CUR(GreedySelector):
    """Transformer that performs Greedy Feature Selection with CUR.

    Refer to :py:class:`skmatter.feature_selection.CUR` for full documentation.
    """

    def __init__(
        self,
        recompute_every=1,
        k=1,
        tolerance=1e-12,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selector_class=_CUR,
            selection_type="feature",
            recompute_every=recompute_every,
            k=k,
            tolerance=tolerance,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )
