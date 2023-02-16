# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Union

import numpy as np
from equistore import Labels, TensorBlock, TensorMap
from equistore.operations import dot, multiply, ones_like, slice
from equistore.operations._utils import _check_blocks, _check_maps

from ...utils.metrics import rmse
from ..utils import block_to_array, dict_to_tensor_map, tensor_map_to_dict


class Ridge:
    r"""Linear least squares with l2 regularization for :class:`equistore.Tensormap`'s.

    Weights :math:`w` are calculated according to

    .. math::

        w = X^T \left( X \cdot X^T + α I \right)^{-1} \cdot y \,,

    where :math:`X` is the training data, :math:`y` the target data and :math:`α` is the
    regularization strength.

    :param parameter_keys: Parameters to perform the regression for.
                           Examples are ``"values"``, ``"positions"``,
                           ``"cell"`` or a combination of these.
    """

    def __init__(
        self,
        parameter_keys: Union[List[str], str] = None,
    ) -> None:
        if parameter_keys is None:
            self.paramater_keys = ["values", "positions"]
        if type(parameter_keys) not in (list, tuple, np.ndarray):
            self.parameter_keys = [parameter_keys]
        else:
            self.parameter_keys = parameter_keys

        self._weights = None

    def _validate_data(self, X: TensorMap, y: Optional[TensorMap] = None) -> None:
        """Validates :class:`equistore.TensorBlock`'s for the usage in models.

        :param X: training data to check
        :param y: target data to check
        """
        if len(X.components_names) != 0:
            raise ValueError("`X` contains components")

        if y is not None:
            _check_maps(X, y, "_validate_data")

            if len(y.components_names) != 0:
                raise ValueError("`y` contains components")

            for key, X_block in X:
                y_block = y.block(key)
                _check_blocks(
                    X_block, y_block, props=["samples"], fname="_validate_data"
                )

                if len(y_block.properties) != 1:
                    raise ValueError(
                        "Only one property is allowed for target values. Given "
                        f"`y` contains {len(y_block.properties)} property."
                    )

    def _validate_params(
        self,
        X: TensorBlock,
        alpha: TensorBlock,
        sample_weight: Optional[TensorBlock] = None,
    ) -> None:
        """Check regulerizer and sample weights have are correct wrt. to``X``.

        :param X: training data for reference
        :param sample_weight: sample weights
        """

        _check_maps(X, alpha, "_validate_params")

        if sample_weight is not None:
            _check_maps(X, sample_weight, "_validate_params")

        for key, X_block in X:
            alpha_block = alpha.block(key)
            _check_blocks(
                X_block, alpha_block, props=["properties"], fname="_validate_params"
            )

            if len(alpha_block.samples) != 1:
                raise ValueError(
                    "Only one sample is allowed for regularization. Given "
                    f"`alpha` contains {len(alpha_block.samples)} samples."
                )

            if sample_weight is not None:
                sw_block = sample_weight.block(key)
                _check_blocks(
                    X_block, sw_block, props=["samples"], fname="_validate_params"
                )

                if len(sw_block.properties) != 1:
                    raise ValueError(
                        "Only one property is allowed for sample weights. Given "
                        f"`sample_weight` contains {len(sw_block.properties)} "
                        "properties."
                    )

    def _numpy_lstsq_solver(self, X, y, sample_weights, alphas, rcond):
        # Convert problem with regularization term into an equivalent
        # problem without the regularization term
        num_properties = X.shape[1]

        regularization_all = np.hstack((sample_weights, alphas))
        regularization_eff = np.diag(np.sqrt(regularization_all))

        X_eff = regularization_eff @ np.vstack((X, np.eye(num_properties)))
        y_eff = regularization_eff @ np.hstack((y, np.zeros(num_properties)))

        return np.linalg.lstsq(X_eff, y_eff, rcond=rcond)[0]

    def fit(
        self,
        X: TensorMap,
        y: TensorMap,
        alpha: Union[float, TensorMap] = 1.0,
        sample_weight: Optional[TensorMap] = None,
        rcond: float = 1e-13,
    ) -> None:
        """Fit Ridge regression model to each block in X.

        :param X: training data
        :param y: target values
        :param alpha: Constant α that multiplies the L2 term, controlling
                      regularization strength. Values must be non-negative floats
                      i.e. in [0, inf). α can be different for each column in ``X``
                      to regulerize each property differently.
        :param sample_weight: sample weights
        :param rcond: Cut-off ratio for small singular values during the fit. For
                    the purposes of rank determination, singular values are treated as
                    zero if they are smaller than ``rcond`` times the largest singular
                    value in "weightsficient" matrix.
        """

        if type(alpha) is float:
            alpha_tensor = ones_like(X)

            samples = Labels(
                names=X.sample_names,
                values=np.zeros([1, len(X.sample_names)], dtype=int),
            )

            alpha_tensor = slice(alpha_tensor, samples=samples)
            alpha = multiply(alpha_tensor, alpha)

        if type(alpha) is not TensorMap:
            raise ValueError("alpha must either be a float or a TensorMap")

        self._validate_data(X, y)
        self._validate_params(X, alpha, sample_weight)

        weights_blocks = []
        for key, X_block in X:
            y_block = y.block(key)
            alpha_block = alpha.block(key)

            # X_arr has shape of (n_targets, n_properties)
            X_arr = block_to_array(X_block, self.parameter_keys)

            # y_arr has shape lentgth of n_targets
            y_arr = block_to_array(y_block, self.parameter_keys)[:, 0]

            # alpha_arr has length of n_properties
            alpha_arr = alpha_block.values[0]

            # Sample weights
            if sample_weight is not None:
                sw_block = sample_weight.block(key)
                # sw_arr has length of n_targets
                sw_arr = block_to_array(sw_block, self.parameter_keys)[:, 0]
                assert (
                    sw_arr.shape == y_arr.shape
                ), f"shapes = {sw_arr.shape} and {y_arr.shape}"
            else:
                sw_arr = np.ones((len(y_arr),))

            w = self._numpy_lstsq_solver(X_arr, y_arr, sw_arr, alpha_arr, rcond)

            weights_block = TensorBlock(
                values=w.reshape(1, -1),
                samples=y_block.properties,
                components=[],
                properties=X_block.properties,
            )
            weights_blocks.append(weights_block)

        # convert weightsficients to a dictionary allowing pickle dump of an instance
        self._weights = tensor_map_to_dict(TensorMap(X.keys, weights_blocks))

        return self

    @property
    def weights(self) -> TensorMap:
        """``Tensormap`` containing the weights of the provided training data."""

        if self._weights is None:
            raise ValueError("No weights. Call fit method first.")

        return dict_to_tensor_map(self._weights)

    def predict(self, X: TensorMap) -> TensorMap:
        """
        Predict using the linear model.

        :param X: samples
        :returns: predicted values
        """
        return dot(X, self.weights)

    def score(self, X: TensorMap, y: TensorMap, parameter_key: str) -> float:
        """Return the weights of determination of the prediction.

        :param X: Test samples
        :param y: True values for ``X``.
        :param parameter_key: Parameter to score for. Examples are ``"values"``,
                              ``"positions"`` or ``"cell"``.

        :returns score: :math:`RMSE` for each block in ``self.predict(X)`` with
                        respecy to `y`.
        """
        if parameter_keys is None:
            parameter_keys = self.parameter_keys
        y_pred = self.predict(X)
        return np.mean([rmse(y, y_pred, parameter_key) for parameter_key in parameter_keys])
