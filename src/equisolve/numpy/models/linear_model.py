# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Union

import numpy as np
from equistore import TensorBlock, TensorMap
from equistore.operations import dot
from equistore.operations._utils import _check_blocks, _check_maps

from ...utils.metrics import rmse
from ..utils import block_to_array, dict_to_tensor_map, tensor_map_to_dict


class Ridge:
    r"""Linear least squares with l2 regularization for :class:`equistore.Tensormap`'s.

    Weights :math:`w` are calculated according to

    .. math::

        w = X^T \left( X \cdot X^T + λ I \right)^{-1} \cdot y \,,

    where :math:`X` is the training data, :math:`y` the target data and :math:`λ` the
    regularization strength.

    :param parameter_keys: Parameters to perform the regression for.
                           Examples are ``"values"``, ``"positions"`` or
                           ``"cell"``.
    :param alpha: Constant :math:`λ` that multiplies the L2 term, controlling
                  regularization strength. Values must be a non-negative floats
                  i.e. in [0, inf). :math:`λ` can be different for each column in ``X``
                  to regulerize each property differently.
    :param rcond: Cut-off ratio for small singular values during the fit. For
                  the purposes of rank determination, singular values are treated as
                  zero if they are smaller than ``rcond`` times the largest singular
                  value in "coefficient" matrix.

    :attr coef_: List :class:`equistore.Tensormap` containing the weights of the
                 provided training data.
    """

    def __init__(
        self,
        parameter_keys: Union[List[str], str] = None,
        alpha: TensorMap = None,
        rcond: float = 1e-13,
    ) -> None:
        if parameter_keys is None:
            paramater_keys = ["values", "positions"]
        if type(parameter_keys) not in (list, tuple, np.ndarray):
            self.parameter_keys = [parameter_keys]
        else:
            self.parameter_keys = parameter_keys


        # TODO(philip) can we make a good default alpha parameter out of paramater_keys?
        if alpha is None:
            raise NotImplemented("Ridge still needs a good default alpha value")
        self.alpha = alpha
        self.rcond = rcond

        self.coef_ = None

        self.alpha = tensor_map_to_dict(self.alpha)

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
        self, X: TensorBlock, sample_weight: Optional[TensorBlock] = None
    ) -> None:
        """Check regulerizer and sample weights have are correct wrt. to``X``.

        :param X: training data for reference
        :param sample_weight: sample weights
        """

        _check_maps(X, self.alpha, "_validate_params")

        if sample_weight is not None:
            _check_maps(X, sample_weight, "_validate_params")

        for key, X_block in X:
            alpha_block = self.alpha.block(key)
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
        self, X: TensorMap, y: TensorMap, sample_weight: Optional[TensorMap] = None
    ) -> None:
        """Fit Ridge regression model to each block in X.

        :param X: training data
        :param y: target values
        :param sample_weight: sample weights
        """
        # If alpha was converted we convert it back here.
        if type(self.alpha) == dict:
            self.alpha = dict_to_tensor_map(self.alpha)

        self._validate_data(X, y)
        self._validate_params(X, sample_weight)

        coef_blocks = []
        for key, X_block in X:
            y_block = y.block(key)
            alpha_block = self.alpha.block(key)

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

            w = self._numpy_lstsq_solver(X_arr, y_arr, sw_arr, alpha_arr, self.rcond)

            coef_block = TensorBlock(
                values=w.reshape(1, -1),
                samples=y_block.properties,
                components=[],
                properties=X_block.properties,
            )
            coef_blocks.append(coef_block)

        self.coef_ = TensorMap(X.keys, coef_blocks)
        # Convert alpha and coefs to a dictionary to be used in external models.
        self.alpha = tensor_map_to_dict(self.alpha)
        self.coef_ = tensor_map_to_dict(self.coef_)

        return self

    def predict(self, X: TensorMap) -> TensorMap:
        """
        Predict using the linear model.

        :param X: samples
        :returns: predicted values
        """
        if self.coef_ is None:
            raise ValueError("No weights. Call fit method first.")

        self.coef_ = dict_to_tensor_map(self.coef_)
        y_pred = dot(X, self.coef_)
        self.coef_ = tensor_map_to_dict(self.coef_)
        return y_pred

    def score(self, X: TensorMap, y: TensorMap, parameter_key: str) -> List[float]: # COMMENT why does it return list of floats if we just allow one paramater_key?
        """Return the coefficient of determination of the prediction.

        :param X: Test samples
        :param y: True values for `X`.
        :param parameter_key: Parameter to score for. Examples are ``"values"``,
                              ``"positions"`` or ``"cell"``.

        :returns score: :math:`RMSE` for each block in ``self.predict(X)`` with
                        respecy to `y`.
        """
        y_pred = self.predict(X)
        return rmse(y, y_pred, parameter_key)