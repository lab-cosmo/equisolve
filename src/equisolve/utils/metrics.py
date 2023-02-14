# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
from typing import List, Callable

import numpy as np

from equistore import TensorMap
from equistore.operations._utils import _check_maps

class Score:
        """Used to compare reduce the output error to one single value
        """
    def __init__(score_function: Callable[[y_true: np.ndarray, y_pred: np.ndarray], np.floating]):
        self._score_function = score_function

    @property
    def score_function(self):
        return self._score_function

    def __call__(y_true: TensorMap, y_pred: TensorMap, *, map_keys=None, parameter_keys: List[str] = None, weight: List[float] = None) -> TensorMap:
        """Mean squared error regression loss.

        :param y_true: Ground truth (correct) target values.
        :param y_pred: Estimated target values.
        :param map_keys: Parameter to perform the score on blocks.
        :param parameter_keys: Parameter to perform the rmse for. Examples are ``"values"``,
                              ``"positions"`` or ``"cell"``. If None, then all parameter_keys in y_pred are used

        :returns loss: A non-negative floating point value (the best value is 0.0), or
                       a tuple of floating point values, one for each block in
                       ``y_pred``.
        """
        _check_maps(y_true, y_pred, "score function")

        loss = []
        for key, y_pred_block in y_pred:
            y_true_block = y_true.block(key)
            _check_blocks(y_true_block, y_pred_block, "score function")
            if (paramater_keys is None) or ("values" in parameter_keys):
                loss_tb = TensorBlock(
                    values=self.score_function(y_true_block.values, y_pred_block.values),
                    samples=Labels.single(), # TODO better name
                    components=[],
                    properties=Labels.single(), # TODO better name
                )
            else:
                loss_tb = TensorBlock(
                    values=np.empty(1,1),
                    samples=Labels.single(), # TODO better name
                    components=[],
                    properties=Labels.single(), # TODO better name
                )


            for paramater_key in paramater_keys:
                if parameter_key == "values":
                    pass
                else:
                    y_pred_values = y_pred_block.gradient(parameter_key).data
                    y_true_values = y_true_block.gradient(parameter_key).data
                    loss_tb.add_gradient(
                            parameter_key,
                            self.score_function(y_true_block.values, y_pred_block.values),
                            samples.
                            components
                    )

                            parameter: str,
                            data: Array,
                            samples: Labels,
                            components: List[Labels],


                y_pred_values = y_pred_values.flatten()
                y_true_values = y_true_values.flatten()

                loss.append(np.sqrt(np.mean((y_pred_values - y_true_values) ** 2)))

        TensorMap(keys=y.keys(), blocks=loss)
        return loss[0]

def rmse(y_true: TensorBlock, y_pred: TensorBlock) -> float:
    return np.sqrt(np.mean((y_pred_values - y_true_values) ** 2))

RMSE = Score(score_function=rmse)

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

    _check_maps(y_true, y_pred, "rmse")

    loss = []
    for key, y_pred_block in y_pred:
        y_true_block = y_true.block(key)

        if parameter_key == "values":
            y_pred_values = y_pred_block.values
            y_true_values = y_true_block.values
        else:
            y_pred_values = y_pred_block.gradient(parameter_key).data
            y_true_values = y_true_block.gradient(parameter_key).data

        y_pred_values = y_pred_values.flatten()
        y_true_values = y_true_values.flatten()

        loss.append(np.sqrt(np.mean((y_pred_values - y_true_values) ** 2)))

    return loss
