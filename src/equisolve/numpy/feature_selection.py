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

    :param initialize:
        Index of the first selection(s). If 'random', picks a random
        value when fit starts.
    :param n_to_select:
        The number of selections to make. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select.
    :param score_threshold:
        Threshold for the score. If `None` selection will continue until the
        `n_to_select` is chosen. Otherwise will stop when the score falls below the
        threshold.
    :param score_threshold_type:
        How to interpret the ``score_threshold``. When "absolute", the score used by the
        selector is compared to the threshold directly. When "relative", at each
        iteration, the score used by the selector is compared proportionally to the
        score of the first selection, i.e. the selector quits when
        ``current_score / first_score < threshold``.
    :param progress_bar:
        Pption to use `tqdm <https://tqdm.github.io/>`_ progress bar to monitor
        selections.
    :param full:
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining features. Stored in :py:attr:`self.full`.
    :param random_state:
        The random state.
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
            selector=_FPS,
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

    The selector is choosing features which maximize the magnitude of the right singular
    vectors, consistent with classic CUR matrix decomposition.

    :param recompute_every:
        Number of steps after which to recompute the pi score
        defaults to 1, if 0 no re-computation is done
    :param k:
        Number of eigenvectors to compute the importance score with, defaults to 1
    :param tolerance:
        Threshold below which scores will be considered 0.
    :param n_to_select:
        The number of selections to make. If `None`, half of the samples are
        selected. If integer, the parameter is the absolute number of selections
        to make. If float between 0 and 1, it is the fraction of the total dataset to
        select.
    :param score_threshold:
        Threshold for the score. If `None` selection will continue until the
        `n_to_select` is chosen. Otherwise will stop when the score falls below the
        threshold.
    :param score_threshold_type:
        How to interpret the ``score_threshold``. When "absolute", the score used by the
        selector is compared to the threshold directly. When "relative", at each
        iteration, the score used by the selector is compared proportionally to the
        score of the first selection, i.e. the selector quits when
        ``current_score / first_score < threshold``.
    :param progress_bar:
        option to use `tqdm <https://tqdm.github.io/>`_
        progress bar to monitor selections.
    :param full:
        In the case that all non-redundant selections are exhausted, choose
        randomly from the remaining samples. Stored in :py:attr:`self.full`.
    :param random_state:
        The random state.
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
            selector=_CUR,
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
