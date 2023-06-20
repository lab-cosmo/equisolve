# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Union

import equistore
import numpy as np
import scipy.linalg
from equistore import Labels, TensorBlock, TensorMap

from ... import HAS_TORCH
from ...module import NumpyModule, _Estimator
from ...utils.metrics import rmse
from ..utils import array_from_block, dict_to_tensor_map, tensor_map_to_dict


class _Ridge(_Estimator):
    r"""Linear least squares with l2 regularization for :class:`equistore.Tensormap`'s.

    Weights :math:`w` are calculated according to

    .. math::

        w = X^T \left( X \cdot X^T + α I \right)^{-1} \cdot y \,,

    where :math:`X` is the training data, :math:`y` the target data and :math:`α` is the
    regularization strength.

    Ridge will regress a model for each block in X. If a block contains components
    the component values will be stacked along the sample dimension for the fit.
    Therefore, the corresponding weights will be the same for each component.
    """

    def __init__(self) -> None:
        self._weights = None

    def _validate_data(self, X: TensorMap, y: Optional[TensorMap] = None) -> None:
        """Validates :class:`equistore.TensorBlock`'s for the usage in models.

        :param X:
            training data to check
        :param y:
            target data to check
        """
        if y is not None and not equistore.equal_metadata(
            X, y, check=["samples", "components"]
        ):
            raise ValueError(
                "Metadata (samples, components) of X and y does not agree!"
            )

    def _validate_params(
        self,
        X: TensorBlock,
        alpha: TensorBlock,
        sample_weight: Optional[TensorBlock] = None,
    ) -> None:
        """Check regulerizer and sample weights have are correct wrt. to ``X``.

        :param X:
            training data for reference
        :param sample_weight:
            sample weights
        """
        if not equistore.equal_metadata(X, alpha, check=["components", "properties"]):
            raise ValueError(
                "Metadata (components, properties) of X and alpha does not agree!"
            )

        if sample_weight is not None and not equistore.equal_metadata(
            X,
            sample_weight,
            check=[
                "samples",
                "components",
            ],
        ):
            raise ValueError(
                "Metadata (samples, components) of X and sample_weight does not agree!"
            )

        for key, alpha_block in alpha:
            if len(alpha_block.samples) != 1:
                raise ValueError(
                    "Only one sample is allowed for regularization. Given "
                    f"`alpha` contains {len(alpha_block.samples)} samples."
                )

            if sample_weight is not None:
                sw_block = sample_weight.block(key)
                if len(sw_block.properties) != 1:
                    raise ValueError(
                        "Only one property is allowed for sample weights. Given "
                        f"`sample_weight` contains {len(sw_block.properties)} "
                        "properties."
                    )

    def _solve(
        self,
        X: TensorBlock,
        y: TensorBlock,
        alpha: TensorBlock,
        sample_weight: TensorBlock,
        cond: float,
    ) -> TensorBlock:
        """A regularized solver using ``np.linalg.lstsq``."""
        self._used_auto_solver = None

        # Convert TensorMaps into arrays for processing them with NumPy.

        # X_arr has shape of (n_targets, n_properties)
        X_arr = array_from_block(X)

        # y_arr has shape lentgth of n_targets
        y_arr = array_from_block(y)

        # sw_arr has shape of (n_samples, 1)
        sw_arr = array_from_block(sample_weight)

        # alpha_arr has shape of (1, n_properties)
        alpha_arr = array_from_block(alpha)

        # Flatten into 1d arrays
        y_arr = y_arr.ravel()
        sw_arr = sw_arr.ravel()
        alpha_arr = alpha_arr.ravel()

        num_properties = X_arr.shape[1]

        sw_arr = sw_arr.reshape(-1, 1)
        sqrt_sw_arr = np.sqrt(sw_arr)

        if self._solver == "auto":
            n_samples, n_features = X_arr.shape

            if n_features > n_samples:
                # solves linear system (K*sw + alpha*Id) @ dual_w = y*sw
                dual_alpha_arr = X_arr @ alpha_arr
                X_eff = sqrt_sw_arr * X_arr
                y_eff = y_arr * sw_arr.flatten()
                K = X_eff @ X_eff.T + np.diag(dual_alpha_arr)
                try:
                    # change to cholesky (issue #36)
                    dual_w = scipy.linalg.solve(K, y_eff, assume_a="pos").ravel()
                    self._used_auto_solver = "cholesky_dual"
                except np.linalg.LinAlgError:
                    # scipy.linalg.pinv default rcond value
                    # https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/linalg/_basic.py#L1341
                    rcond = max(X_arr.shape) * np.finfo(X_arr.dtype.char.lower()).eps
                    dual_w = scipy.linalg.lstsq(K, y_eff, cond=rcond, overwrite_a=True)[
                        0
                    ].ravel()
                    self._used_auto_solver = "lstsq_dual"
                w = X_arr.T @ dual_w
            else:
                XtX = X_arr.T @ (sw_arr * X_arr) + np.diag(alpha_arr)
                Xt_y = X_arr.T @ (sw_arr.flatten() * y_arr)
                try:
                    # change to cholesky (issue #36)
                    # solves linear system (X.T @ X * sw + alpha*Id) @ w = X.T @ sw*y
                    w = scipy.linalg.solve(XtX, Xt_y, assume_a="pos")
                    self._used_auto_solver = "cholesky"
                except np.linalg.LinAlgError:
                    s, U = scipy.linalg.eigh(XtX)
                    # scipy.linalg.pinv default rcond value
                    # https://github.com/scipy/scipy/blob/c1ed5ece8ffbf05356a22a8106affcd11bd3aee0/scipy/linalg/_basic.py#L1341
                    rcond = max(X_arr.shape) * np.finfo(X_arr.dtype.char.lower()).eps
                    eigval_cutoff = np.sqrt(s[-1]) * rcond
                    n_dim = sum(s > eigval_cutoff)
                    w = (U[:, -n_dim:] / s[-n_dim:]) @ U[:, -n_dim:].T @ Xt_y
                    w = w.ravel()
                    self._used_auto_solver = "svd_primal"
        elif self._solver == "cholesky":
            # solves linear system (X.T @ X * sw + alpha*Id) @ w = X.T @ sw*y
            X_eff = np.vstack([sqrt_sw_arr * X_arr, np.diag(np.sqrt(alpha_arr))])
            y_eff = np.hstack([y_arr * sqrt_sw_arr.flatten(), np.zeros(num_properties)])
            XXt = X_eff.T @ X_eff
            Xt_y = X_eff.T @ y_eff
            # change to cholesky (issue #36)
            w = scipy.linalg.solve(XXt, Xt_y, assume_a="pos")
        elif self._solver == "cholesky_dual":
            # solves linear system (K*sw + alpha*Id) @ dual_w = y*sw
            dual_alpha_arr = X_arr @ alpha_arr
            X_eff = sqrt_sw_arr * X_arr
            y_eff = y_arr * sw_arr.flatten()
            K = X_eff @ X_eff.T + np.diag(dual_alpha_arr)
            # change to cholesky (issue #36)
            dual_w = scipy.linalg.solve(K, y_eff, assume_a="pos").ravel()
            w = X_arr.T @ dual_w
        elif self._solver == "lstsq":
            # We solve system of form Ax=b, where A is [X*sqrt(sw), Id*sqrt(alpha)]
            # and b is [y*sqrt(w), 0]
            X_eff = np.vstack([sqrt_sw_arr * X_arr, np.diag(np.sqrt(alpha_arr))])
            y_eff = np.hstack([y_arr * sqrt_sw_arr.flatten(), np.zeros(num_properties)])
            w = scipy.linalg.lstsq(X_eff, y_eff, cond=cond, overwrite_a=True)[0].ravel()
        else:
            raise ValueError(
                f"Unknown solver {self._solver} only 'auto', 'cholesky',"
                " 'cholesky_dual' and 'lstsq' are supported."
            )

        # Reshape values into 1 sample + component shape + num_properties.
        # The weights will the same for each component.
        shape = [1] + [len(c) for c in X.components] + [num_properties]
        values = np.tile(w, np.prod(shape[:-1])).reshape(shape)

        weights_block = TensorBlock(
            values=values,
            samples=y.properties,
            components=X.components,
            properties=X.properties,
        )

        return weights_block

    def fit(
        self,
        X: TensorMap,
        y: TensorMap,
        alpha: Union[float, TensorMap] = 1.0,
        sample_weight: Union[float, TensorMap] = None,
        solver="auto",
        cond: float = None,
    ) -> None:
        """Fit a regression model to each block in `X`.

        Ridge takes all available values and gradients in the provided TensorMap for the
        fit. Gradients can be exlcuded from the fit if removed from the TensorMap.
        See :py:func:`equistore.remove_gradients` for details.

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
        :param solver:
            Solver to use in the computational routines:

            - **"auto"**: If n_features > n_samples in X, it solves the dual problem
              using cholesky_dual.
              If this fails, it switches to :func:`scipy.linalg.lstsq` to solve
              the dual problem.
              If n_features <= n_samples in X, it solves the primal problem
              using cholesky.
              If this fails, it switches to :func:`scipy.linalg.eigh` on (X.T @ X)
              and cuts off eigenvalues
              below machine precision times maximal shape of X.
            - **"cholesky"**: using :func:`scipy.linalg.solve` on (X.T@X) w = X.T @ y
            - **"cholesky_dual"**: using :func:`scipy.linalg.solve`
              on the dual problem (X@X.T) w_dual = y,
              the primal weights are obtained by w = X.T @ w_dual
            - **"lstsq"**: using :func:`scipy.linalg.lstsq` on the linear system X w = y
        :param cond:
            Cut-off ratio for small singular values during the fit. For the purposes of
            rank determination, singular values are treated as zero if they are smaller
            than `cond` times the largest singular value in "weights" matrix.
        """
        self._solver = solver

        if type(alpha) is float:
            alpha_tensor = equistore.ones_like(X)

            samples = Labels(
                names=X.sample_names,
                values=np.zeros([1, len(X.sample_names)], dtype=int),
            )

            alpha_tensor = equistore.slice(alpha_tensor, axis="samples", labels=samples)
            alpha = equistore.multiply(alpha_tensor, alpha)
        elif type(alpha) is not TensorMap:
            raise ValueError("alpha must either be a float or a TensorMap")

        if sample_weight is None:
            sample_weight = equistore.ones_like(y)
        elif type(sample_weight) is float:
            sample_weight = equistore.multiply(equistore.ones_like(y), sample_weight)
        elif type(sample_weight) is not TensorMap:
            raise ValueError("sample_weight must either be a float or a TensorMap.")

        self._validate_data(X, y)
        self._validate_params(X, alpha, sample_weight)

        # Remove all gradients here. This is a workaround until we can exclude gradients
        # for metadata check (see equistore issue #285 for details)
        alpha = equistore.remove_gradients(alpha)

        weights_blocks = []
        for key, X_block in X:
            y_block = y.block(key)
            alpha_block = alpha.block(key)
            sw_block = sample_weight.block(key)

            weight_block = self._solve(X_block, y_block, alpha_block, sw_block, cond)

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
        return equistore.dot(X, self.weights)

    def forward(self, X: TensorMap) -> TensorMap:
        return self.predict(X)

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


class NumpyRidge(_Ridge, NumpyModule):
    def __init__(self) -> None:
        NumpyModule.__init__(self)
        _Ridge.__init__(self)


if HAS_TORCH:
    import torch

    class TorchRidge(_Ridge, torch.nn.Module):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            _Ridge.__init__(self)

    Ridge = TorchRidge
else:
    Ridge = NumpyRidge
