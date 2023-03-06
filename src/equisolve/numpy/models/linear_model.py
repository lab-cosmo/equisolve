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

    :param parameter_keys:
        Parameters to perform the regression for. Examples are ``"values"``,
        ``"positions"``, ``"cell"`` or a combination of these.
    """

    def __init__(
        self,
        parameter_keys: Union[List[str], str] = None,
    ) -> None:
        if type(parameter_keys) not in (list, tuple, np.ndarray):
            self.parameter_keys = [parameter_keys]
        else:
            self.parameter_keys = parameter_keys

        self._weights = None

    def _validate_data(self, X: TensorMap, y: Optional[TensorMap] = None) -> None:
        """Validates :class:`equistore.TensorBlock`'s for the usage in models.

        :param X:
            training data to check
        :param y:
            target data to check
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

        :param X:
            training data for reference
        :param sample_weight:
            sample weights
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

    def _solver(
        self,
        X: TensorBlock,
        y: TensorBlock,
        alpha: TensorBlock,
        sample_weight: TensorBlock,
        rcond: float,
    ) -> TensorBlock:
        """A regularized solver using ``np.linalg.lstsq``."""

        # Convert TensorMaps into arrays for processing them with NumPy.

        # X_arr has shape of (n_targets, n_properties)
        X_arr = block_to_array(X, self.parameter_keys)

        # y_arr has shape lentgth of n_targets
        y_arr = block_to_array(y, self.parameter_keys)

        # sw_arr has shape of (n_samples, 1)
        sw_arr = block_to_array(sample_weight, self.parameter_keys)

        # alpha_arr has shape of (1, n_properties)
        alpha_arr = block_to_array(alpha, ["values"])

        # Flatten into 1d arrays
        y_arr = y_arr.ravel()
        sw_arr = sw_arr.ravel()
        alpha_arr = alpha_arr.ravel()

        # Convert problem with regularization term into an equivalent
        # problem without the regularization term
        num_properties = X_arr.shape[1]

        regularization_all = np.hstack((sw_arr, alpha_arr))
        regularization_eff = np.diag(np.sqrt(regularization_all))

        X_eff = regularization_eff @ np.vstack((X_arr, np.eye(num_properties)))
        y_eff = regularization_eff @ np.hstack((y_arr, np.zeros(num_properties)))

        w = np.linalg.lstsq(X_eff, y_eff, rcond=rcond)[0]

        weights_block = TensorBlock(
            values=w.reshape(1, -1),
            samples=y.properties,
            components=[],
            properties=X.properties,
        )

        return weights_block

    def fit(
        self,
        X: TensorMap,
        y: TensorMap,
        alpha: Union[float, TensorMap] = 1.0,
        sample_weight: Union[float, TensorMap] = None,
        rcond: float = 1e-13,
    ) -> None:
        """Fit a regression model to each block in `X`.

        :param X:
            training data
        :param y:
            target values
        :param alpha:
            Constant α that multiplies the L2 term, controlling regularization strength.
            Values must be non-negative floats i.e. in [0, inf). α can be different for
            each column in `X` to regulerize each property differently.
        :param sample_weight:
            Individual weights for each sample. For `None` or a float, every sample will
            have the same weight of 1 or the float, respectively..
        :param rcond:
            Cut-off ratio for small singular values during the fit. For the purposes of
            rank determination, singular values are treated as zero if they are smaller
            than `rcond` times the largest singular value in "weights" matrix.
        """

        if type(alpha) is float:
            alpha_tensor = ones_like(X)

            samples = Labels(
                names=X.sample_names,
                values=np.zeros([1, len(X.sample_names)], dtype=int),
            )

            alpha_tensor = slice(alpha_tensor, samples=samples)
            alpha = multiply(alpha_tensor, alpha)
        elif type(alpha) is not TensorMap:
            raise ValueError("alpha must either be a float or a TensorMap")

        if sample_weight is None:
            sample_weight = ones_like(y)
        elif type(sample_weight) is float:
            sample_weight = multiply(ones_like(y), sample_weight)
        elif type(sample_weight) is not TensorMap:
            raise ValueError("sample_weight must either be a float or a TensorMap.")

        self._validate_data(X, y)
        self._validate_params(X, alpha, sample_weight)

        weights_blocks = []
        for key, X_block in X:
            y_block = y.block(key)
            alpha_block = alpha.block(key)
            sw_block = sample_weight.block(key)

            weight_block = self._solver(X_block, y_block, alpha_block, sw_block, rcond)

            weights_blocks.append(weight_block)

        # convert weights to a dictionary allowing pickle dump of an instance
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

        :param X:
            samples
        :returns:
            predicted values
        """
        return dot(X, self.weights)

    def score(self, X: TensorMap, y: TensorMap, parameter_key: str) -> float:
        r"""Return the weights of determination of the prediction.

        :param X:
            Test samples
        :param y:
            True values for ``X``.
        :param parameter_key:
            Parameter to score for. Examples are ``"values"``, ``"positions"`` or
            ``"cell"``.

        :returns score:
            :math:`\mathrm{RMSE}` for each block in ``self.predict(X)`` with respecy to
            `y`.
        """
        y_pred = self.predict(X)
        return rmse(y, y_pred, parameter_key)
