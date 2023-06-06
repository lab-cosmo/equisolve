# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Union, Type

import equistore
from equistore import Labels, TensorBlock, TensorMap
import numpy as np
import scipy.linalg

from ... import HAS_TORCH
from ...module import NumpyModule, _Estimator
from ...utils.metrics import rmse
from ..utils import block_to_array, dict_to_tensor_map, tensor_map_to_dict

if HAS_TORCH:
    from equistore.torch import Labels as TorchLabels, TensorBlock as TorchTensorBlock, TensorMap as TorchTensorMap



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

        weights_blocks = []
        for key, X_block in X:
            y_block = y.block(key)
            alpha_block = alpha.block(key)
            sw_block = sample_weight.block(key)

            weight_block = self._solve(X_block, y_block, alpha_block, sw_block, cond)

            weights_blocks.append(weight_block)

        # convert weights to a dictionary allowing pickle dump of an instance
        self._set_weights(TensorMap(X.keys, weights_blocks))
        return self

    def _set_weights(self, weights: TensorMap) -> None:
        self._weights = weights

    # @property is not supported by torchscript
    def weights(self) -> TensorMap:
        """``Tensormap`` containing the weights of the provided training data."""

        if self._weights is None:
            raise ValueError("No weights. Call fit method first.")

        return self._weights

    def predict(self, X: TensorMap) -> TensorMap:
        """
        Predict using the linear model.

        :param X:
            samples
        :returns:
            predicted values
        """
        return equistore.dot(X, self.weights())

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
    def __init__(
        self,
        parameter_keys: Union[List[str], str] = None,
    ) -> None:
        NumpyModule.__init__(self)
        _Ridge.__init__(self, parameter_keys)


if HAS_TORCH:
    import torch

    class TorchRidge(_Ridge, torch.nn.Module):
        def __init__(
            self,
            parameter_keys: Union[List[str], str] = None,
        ) -> None:
            torch.nn.Module.__init__(self)
            _Ridge.__init__(self, parameter_keys)

        def fit(
            self,
            X: TensorMap,
            y: TensorMap,
            alpha: Union[float, TensorMap] = 1.0,
            sample_weight: Union[float, TensorMap] = None,
            solver="auto",
            cond: float = None,
        ) -> None:
            _Ridge.fit(self, X, y, alpha, sample_weight, solver, cond)

        # Required because in torch.nn.Module we cannot reasign variables
        # Once set we can only update the values 
        def _set_weights(self, weights: TorchTensorMap) -> None:
            self._weights = to_torch_tensor_map(weights)

        def predict(self, X: TorchTensorMap) -> TorchTensorMap:
            return dot(X, self.weights())

        def forward(self, X: TorchTensorMap) -> TorchTensorMap:
            return self.predict(X)

        def weights(self) -> TorchTensorMap:
            """``Tensormap`` containing the weights of the provided training data."""

            if self._weights is None:
                raise ValueError("No weights. Call fit method first.")

            return self._weights

    Ridge = TorchRidge
else:
    Ridge = NumpyRidge


def to_torch_labels(labels : Labels):
    return TorchLabels(
        names = list(labels.dtype.names),
        values = torch.tensor(labels.tolist(), dtype=torch.int32)
    )
def to_torch_tensor_map(tensor_map : TensorMap):
    blocks = []
    for _, block in tensor_map:
        blocks.append(TorchTensorBlock(
                values = torch.tensor(block.values),
                samples = to_torch_labels(block.samples),
                components = [to_torch_labels(component) for component in block.components],
                properties = to_torch_labels(block.properties),
            )
        )

    return TorchTensorMap(
            keys = to_torch_labels(tensor_map.keys),
            blocks = blocks,
        )


def dot(tensor_1: TorchTensorMap, tensor_2: TorchTensorMap) -> TorchTensorMap:
    """Compute the dot product of two :py:class:`TensorMap`.
    """
    _check_same_keys(tensor_1, tensor_2, "dot")

    blocks: List[TorchTensorBlock] = []
    for key, block_1 in tensor_1.items():
        block_2 = tensor_2.block(key)
        blocks.append(_dot_block(block_1=block_1, block_2=block_2))

    return TorchTensorMap(tensor_1.keys, blocks)


def _dot_block(block_1: TorchTensorBlock, block_2: TorchTensorBlock) -> TorchTensorBlock:
    if not torch.all(torch.tensor(block_1.properties == block_2.properties)):
        raise ValueError("TensorBlocks in `dot` should have the same properties")

    if len(block_2.components) > 0:
        raise ValueError("the second TensorMap in `dot` should not have components")

    if len(block_2.gradients_list()) > 0:
        raise ValueError("the second TensorMap in `dot` should not have gradients")

    # values = block_1.values @ block_2.values.T
    values = _dispatch_dot(block_1.values, block_2.values)

    result_block = TorchTensorBlock(
        values=values,
        samples=block_1.samples,
        components=block_1.components,
        properties=block_2.samples,
    )

    # I dont know wtf torchscript wants here
    for parameter, gradient in block_1.gradients().items():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        # gradient_values = gradient.values @ block_2.values.T
        gradient_values = _dispatch_dot(gradient.values, block_2.values)

        result_block.add_gradient(
            parameter=parameter,
            gradient=TorchTensorBlock(
                values=gradient_values,
                samples=gradient.samples,
                components=gradient.components,
                properties=result_block.properties,
            ),
        )

    return result_block

def _check_same_keys(a: TorchTensorMap, b: TorchTensorMap, fname: str):
    """Check if metadata between two TensorMaps is consistent for an operation.

    The functions verifies that

    1. The key names are the same.
    2. The number of blocks in the same
    3. The block key indices are the same.

    :param a: first :py:class:`TensorMap` for check
    :param b: second :py:class:`TensorMap` for check
    """

    keys_a: TorchLabels = a.keys
    keys_b: TorchLabels = b.keys

    if keys_a.names != keys_b.names:
        raise ValueError(
            f"inputs to {fname} should have the same keys names, "
            f"got '{keys_a.names}' and '{keys_b.names}'"
        )

    if len(keys_a) != len(keys_b):
        raise ValueError(
            f"inputs to {fname} should have the same number of blocks, "
            f"got {len(keys_a)} and {len(keys_b)}"
        )

    #list_keys: List[bool] = []
    #for i in range(len(keys_b)):
    #    is_in_key_a = keys_b[i] in keys_a
    #for key in keys_b:
    #    is_in_key_a = key in keys_a
    #    list_keys.append(is_in_key_a)

    ##if not torch.all(torch.tensor(list_keys)):
    if not torch.all(torch.tensor([keys_b[i] in keys_a for i in range(len(keys_b))])):
        raise ValueError(f"inputs to {fname} should have the same keys")


def _check_blocks(a: TensorBlock, b: TensorBlock, props: List[str], fname: str):
    """Check if metadata between two TensorBlocks is consistent for an operation.

    The functions verifies that that the metadata of the given props is the same
    (length and indices).

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param props: A list of strings containing the property to check.
                 Allowed values are ``'properties'`` or ``'samples'``,
                 ``'components'`` and ``'gradients'``.
    """
    for prop in props:
        err_msg = f"inputs to '{fname}' should have the same {prop}:\n"
        err_msg_len = f"{prop} of the two `TensorBlock` have different lengths"
        err_msg_1 = f"{prop} are not the same or not in the same order"
        err_msg_names = f"{prop} names are not the same or not in the same order"

        if prop == "samples":
            if not len(a.samples) == len(b.samples):
                raise ValueError(err_msg + err_msg_len)
            if not a.samples.names == b.samples.names:
                raise ValueError(err_msg + err_msg_names)
            if not torch.all(a.samples == b.samples):
                raise ValueError(err_msg + err_msg_1)

        elif prop == "properties":
            if not len(a.properties) == len(b.properties):
                raise ValueError(err_msg + err_msg_len)
            if not a.properties.names == b.properties.names:
                raise ValueError(err_msg + err_msg_names)
            if not torch.all(torch.tensor(a.properties == b.properties)):
                raise ValueError(err_msg + err_msg_1)

        elif prop == "components":
            if len(a.components) != len(b.components):
                raise ValueError(err_msg + err_msg_len)

            for c1, c2 in zip(a.components, b.components):
                if not (c1.names == c2.names):
                    raise ValueError(err_msg + err_msg_names)

                if not (len(c1) == len(c2)):
                    raise ValueError(err_msg + err_msg_1)

                if not torch.all(c1 == c2):
                    raise ValueError(err_msg + err_msg_1)

        else:
            raise ValueError(
                f"{prop} is not a valid property to check, "
                "choose from ['samples', 'properties', 'components']"
            )


def _dispatch_dot(A, B):
    """Compute dot product of two arrays.

    This function has the same behavior as  ``np.dot(A, B.T)``, and assumes the
    second array is 2-dimensional.
    """
    #if isinstance(A, np.ndarray):
    #    _check_all_same_type([B], np.ndarray)
    #    shape1 = A.shape
    #    assert len(B.shape) == 2
    #    # Using matmul/@ is the recommended way in numpy docs for 2-dimensional
    #    # matrices
    #    if len(shape1) == 2:
    #        return A @ B.T
    #    else:
    #        return np.dot(A, B.T)
    if isinstance(A, torch.Tensor):
        assert len(B.shape) == 2
        return A @ B.T
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)
