from typing import Optional

import equistore
import numpy as np
from equistore import Labels, TensorBlock, TensorMap

from ... import HAS_TORCH
from ...module import NumpyModule, _Transformer
from ..utils import array_from_block, dict_to_tensor_map, tensor_map_to_dict


class _StandardScaler(_Transformer):
    """Standardize features by removing the mean and scaling to unit variance.

    :param with_mean:
        If ``True``, center the data before scaling. If ``False``,
        keep the mean intact. Because all operations are consistent
        wrt. to the derivative, the mean is only taken from the values,
        but not from the gradients, since ∇(X - mean) = ∇X
    :param with_std:
        If ``True``, scale the data to unit variance. If ``False``,
        keep the variance intact
    :param column_wise:
        If True, normalize each column separately. If False, normalize the whole
        matrix with respect to its total variance.
    :param rtol:
        The relative tolerance for the optimization: variance is
        considered zero when it is less than abs(mean) * rtol + atol.
    :param atol:
        The relative tolerance for the optimization: variance is
        considered zero when it is less than abs(mean) * rtol + atol.
    """

    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        column_wise: bool = False,
        rtol: float = 0.0,
        atol: float = 1e-12,
    ):
        self.with_mean = with_mean
        self.with_std = with_std
        self.column_wise = column_wise
        self.rtol = rtol
        self.atol = atol

    def _validate_data(self, X: TensorMap, y: TensorMap = None):
        """Validates :class:`equistore.TensorBlock`'s for the usage in models.

        :param X: training data to check
        :param y: target data to check
        """
        if len(X.components_names) != 0:
            raise ValueError("X contains components")

        if y is not None:
            if not equistore.equal_metadata(X, y, check=["samples"]):
                raise ValueError(
                    "Metadata (samples) of X and sample_weight does not agree!"
                )

            if len(y.components_names) != 0:
                raise ValueError("y contains components")

    def fit(
        self,
        X: TensorMap,
        y: TensorMap = None,
        sample_weights: Optional[TensorMap] = None,
    ):
        # y is accepted as None for later pipelines
        self._validate_data(X, y)

        for X_block in X.blocks():
            n_components_block = TensorBlock(
                values=np.array([len(X_block.components)], dtype=np.int32).reshape(
                    1, 1
                ),
                samples=Labels.single(),
                components=[],
                properties=Labels.single(),
            )

        # TODO use this in transform function to do a check
        self.n_components_ = TensorMap(X.keys, [n_components_block])
        self.n_properties_ = len(X_block.properties)

        mean_blocks = []
        scale_blocks = []

        # replace with equistore oprations see issue #18
        for key, X_block in X.items():
            # if values not in parameter_keys, we create empty tensor block to
            # attach gradients
            X_mat = X_block.values

            if sample_weights is not None:
                sw_block = sample_weights.block(key)
                sample_weights = sw_block.values

            if self.with_mean:
                mean_values = np.average(X_mat, weights=sample_weights, axis=0)
            else:
                mean_values = np.zeros(self.n_properties_)

            mean_block = TensorBlock(
                values=mean_values.reshape((1,) + mean_values.shape),
                samples=Labels(["sample"], np.array([[0]], dtype=np.int32)),
                components=[],
                properties=X_block.properties,
            )

            if self.with_std:
                X_mean = np.average(X_mat, weights=sample_weights, axis=0)
                var = np.average((X_mat - X_mean) ** 2, axis=0)

                if self.column_wise:
                    if np.any(var < self.atol + abs(X_mean) * self.rtol):
                        raise ValueError(
                            "Cannot normalize a property with zero variance"
                        )
                    scale_values = np.sqrt(var).reshape(1, var.shape[0])
                else:
                    var_sum = var.sum()
                    if var_sum < abs(np.mean(X_mean)) * self.rtol + self.atol:
                        raise ValueError("Cannot normalize a matrix with zero variance")
                    scale_values = np.sqrt(var_sum).reshape(1, 1)
            else:
                scale_values = np.ones((1, 1))

            scale_block = TensorBlock(
                values=scale_values.reshape((1, 1)),
                samples=Labels.single(),
                components=[],
                properties=Labels.single(),
            )

            for parameter in X_block.gradients_list():
                values = X_block.gradient(parameter).values
                X_mat = values.reshape(np.prod(values.shape[:-1]), values.shape[-1])

                if sample_weights is not None:
                    sw_block = sample_weights.block(key)
                    sample_weights = array_from_block(sw_block, [parameter])
                if self.with_mean:
                    mean_values = np.average(X_mat, weights=sample_weights, axis=0)
                    mean_values = mean_values.reshape((1, 1) + mean_values.shape)
                else:
                    mean_values = np.zeros((1, 1, self.n_properties_))

                gradient = TensorBlock(
                    values=mean_values.reshape(1, 1, -1),
                    samples=Labels(["sample"], np.array([[0]], dtype=np.int32)),
                    components=[Labels.single()],
                    properties=mean_block.properties,
                )
                mean_block.add_gradient(parameter, gradient)

                if self.with_std:
                    X_mean = np.average(X_mat, weights=sample_weights, axis=0)
                    var = np.average((X_mat - X_mean) ** 2, axis=0)

                    if self.column_wise:
                        if np.any(var < self.atol + abs(X_mean) * self.rtol):
                            raise ValueError(
                                "Cannot normalize a property with zero variance"
                            )
                        scale_values = np.sqrt(var).reshape(1, var.shape[0])
                    else:
                        var_sum = var.sum()
                        if var_sum < abs(np.mean(X_mean)) * self.rtol + self.atol:
                            raise ValueError(
                                "Cannot normalize a matrix with zero variance"
                            )
                        scale_values = np.sqrt(var_sum).reshape(1, 1)
                else:
                    scale_values = np.ones((1, 1))
                gradient = TensorBlock(
                    values=scale_values,
                    samples=Labels(["sample"], np.array([[0]])),
                    components=scale_block.components,
                    properties=scale_block.properties,
                )
                scale_block.add_gradient(parameter, gradient)

            mean_blocks.append(mean_block)
            scale_blocks.append(scale_block)

        self.mean_map_ = TensorMap(X.keys, mean_blocks)
        self.scale_map_ = TensorMap(X.keys, scale_blocks)

        self.mean_map_ = tensor_map_to_dict(self.mean_map_)
        self.scale_map_ = tensor_map_to_dict(self.scale_map_)
        # TODO need a tensor_block_to_dict function
        #      atm we dont need them after the fit so we set them to None
        self.n_properties_ = None
        self.n_components_ = None

        return self

    def transform(self, X: TensorMap, y: TensorMap = None):
        # TODO general is_fitted check function
        if not (hasattr(self, "scale_map_")):
            raise ValueError("No scale_map_. Call fit method first")
        if not (hasattr(self, "mean_map_")):
            raise ValueError("No mean_map_. Call fit method first")

        self.mean_map_ = dict_to_tensor_map(self.mean_map_)
        self.scale_map_ = dict_to_tensor_map(self.scale_map_)

        if len(X.blocks()) != len(self.mean_map_):
            raise ValueError(
                f"Number of blocks in X ({len(X.blocks())}) does not agree "
                f"with the number of fitted means ({len(self.mean_map_.blocks())})."
            )

        if len(X.blocks()) != len(self.scale_map_):
            raise ValueError(
                f"Number of blocks in X ({len(X.blocks())}) does not agree "
                f"with the number of fitted scales ({len(self.scale_map_.blocks())})."
            )

        blocks = []
        for key, X_block in X.items():
            mean_block = self.mean_map_.block(key)
            scale_block = self.scale_map_.block(key)

            block = TensorBlock(
                values=(X_block.values - mean_block.values) / scale_block.values,
                samples=X_block.samples,
                components=[],
                properties=X_block.properties,
            )

            for parameter in X_block.gradients_list():
                X_gradient = X_block.gradient(parameter)

                gradient = TensorBlock(
                    values=(X_gradient.values - mean_block.gradient(parameter).values)
                    / scale_block.gradient(parameter).values,
                    samples=X_gradient.samples,
                    components=X_gradient.components,
                    properties=X_gradient.properties,
                )

                block.add_gradient(parameter, gradient)
            blocks.append(block)

        self.mean_map_ = tensor_map_to_dict(self.mean_map_)
        self.scale_map_ = tensor_map_to_dict(self.scale_map_)
        return TensorMap(X.keys, blocks)

    def inverse_transform(self, X: TensorMap, y: TensorMap = None):
        # TODO general is_fitted check function
        if not (hasattr(self, "scale_map_")):
            raise ValueError("No scale_map_. Call fit method first")
        if not (hasattr(self, "mean_map_")):
            raise ValueError("No mean_map_. Call fit method first")

        self.mean_map_ = dict_to_tensor_map(self.mean_map_)
        self.scale_map_ = dict_to_tensor_map(self.scale_map_)

        if len(X.blocks()) != len(self.mean_map_):
            raise ValueError(
                f"Number of blocks in X ({len(X.blocks())}) does not agree "
                f"with the number of fitted means ({len(self.mean_map_.blocks())})."
            )

        if len(X.blocks()) != len(self.scale_map_):
            raise ValueError(
                f"Number of blocks in X ({len(X.blocks())}) does not agree "
                f"with the number of fitted scales ({len(self.scale_map_.blocks())})."
            )

        blocks = []
        for key, X_block in X.items():
            mean_block = self.mean_map_.block(key)
            scale_block = self.scale_map_.block(key)

            block = TensorBlock(
                values=(X_block.values * scale_block.values) + mean_block.values,
                samples=X_block.samples,
                components=[],
                properties=X_block.properties,
            )

            for parameter in X_block.gradients_list():
                X_gradient = X_block.gradient(parameter)

                gradient = TensorBlock(
                    values=(X_gradient.values * scale_block.gradient(parameter).values)
                    + mean_block.gradient(parameter).values,
                    samples=X_gradient.samples,
                    components=X_gradient.components,
                    properties=X_gradient.properties,
                )
                block.add_gradient(parameter, gradient)
            blocks.append(block)

        self.mean_map_ = tensor_map_to_dict(self.mean_map_)
        self.scale_map_ = tensor_map_to_dict(self.scale_map_)
        return TensorMap(X.keys, blocks)


class NumpyStandardScaler(_StandardScaler, NumpyModule):
    def __init__(
        self,
        with_mean: bool = True,
        with_std: bool = True,
        column_wise: bool = False,
        rtol: float = 0.0,
        atol: float = 1e-12,
    ) -> None:
        NumpyModule.__init__(self)
        _StandardScaler.__init__(self, with_mean, with_std, column_wise, rtol, atol)


if HAS_TORCH:
    import torch

    class TorchStandardScaler(_StandardScaler, torch.nn.Module):
        def __init__(
            self,
            with_mean: bool = True,
            with_std: bool = True,
            column_wise: bool = False,
            rtol: float = 0.0,
            atol: float = 1e-12,
        ) -> None:
            torch.nn.Module.__init__(self)
            _StandardScaler.__init__(self, with_mean, with_std, column_wise, rtol, atol)

    StandardScaler = TorchStandardScaler
else:
    StandardScaler = NumpyStandardScaler
