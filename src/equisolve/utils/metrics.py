# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
"""Functions for evaluating the quality of a model’s predictions."""

from typing import List

import equistore
import numpy as np
from equistore import TensorMap


def rmse(y_true: TensorMap, y_pred: TensorMap, parameter_key: str) -> List[float]:
    """Mean squared error regression loss.

    :param y_true: Ground truth (correct) target values.
    :param y_pred: Estimated target values.
    :param parameter_key: Parameter to perform the rmse for. Examples are ``"values"``,
                          ``"positions"`` or ``"cell"``.

    :returns loss: A non-negative floating point value (the best value is 0.0), or
                   a tuple of floating point values, one for each block in
                   ``y_pred``.
    """

    if not equistore.equal_metadata(y_true, y_pred):
        raise ValueError("Metadata of X and sample_weight does not agree!")

    loss = []
    for key, y_pred_block in y_pred:
        y_true_block = y_true.block(key)

        if parameter_key == "values":
            y_pred_values = y_pred_block.values
            y_true_values = y_true_block.values
        else:
            y_pred_values = y_pred_block.gradient(parameter_key).values
            y_true_values = y_true_block.gradient(parameter_key).values

        y_pred_values = y_pred_values.flatten()
        y_true_values = y_true_values.flatten()

        loss.append(np.sqrt(np.mean((y_pred_values - y_true_values) ** 2)))

    return loss
