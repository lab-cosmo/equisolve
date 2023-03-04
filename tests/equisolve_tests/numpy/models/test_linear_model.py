# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from equistore import Labels, TensorBlock, TensorMap
from numpy.testing import assert_allclose, assert_equal

from equisolve.numpy.models import Ridge
from equisolve.numpy.utils import matrix_to_block, tensor_to_tensormap


def numpy_solver(X, y, sample_weights, regularizations):
    _, num_properties = X.shape

    # Convert problem with regularization term into an equivalent
    # problem without the regularization term
    regularization_all = np.hstack((sample_weights, regularizations))
    regularization_eff = np.diag(np.sqrt(regularization_all))
    X_eff = regularization_eff @ np.vstack((X, np.eye(num_properties)))
    y_eff = regularization_eff @ np.hstack((y, np.zeros(num_properties)))
    least_squares_output = np.linalg.lstsq(X_eff, y_eff, rcond=1e-13)
    w_solver = least_squares_output[0]

    return w_solver


class TestRidge:
    """Test the class for a linear ridge regression."""

    @pytest.fixture(autouse=True)
    def set_random_generator(self):
        """Set the random generator to same seed before each test is run.
        Otherwise test behaviour is dependend on the order of the tests
        in this file and the number of parameters of the test.
        """
        self.rng = np.random.default_rng(0x1225787418FBDFD12)

    def to_equistore(self, X_arr=None, y_arr=None, alpha_arr=None, sw_arr=None):
        """Convert Ridge parameters into equistore Tensormap's with one block."""

        returns = ()

        if X_arr is not None:
            assert len(X_arr.shape) == 2
            X = tensor_to_tensormap(X_arr[np.newaxis, :])
            returns += (X,)
        if y_arr is not None:
            assert len(y_arr.shape) == 1
            y = tensor_to_tensormap(y_arr.reshape(1, -1, 1))
            returns += (y,)
        if alpha_arr is not None:
            assert len(alpha_arr.shape) == 1
            alpha = tensor_to_tensormap(alpha_arr.reshape(1, 1, -1))
            returns += (alpha,)
        if sw_arr is not None:
            assert len(sw_arr.shape) == 1
            sw = tensor_to_tensormap(sw_arr.reshape(1, -1, 1))
            returns += (sw,)

        if len(returns) == 0:
            return None
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

    def equisolve_solver_from_numpy_arrays(self, X_arr, y_arr, alpha_arr, sw_arr=None):
        X, y, alpha, sw = self.to_equistore(X_arr, y_arr, alpha_arr, sw_arr)
        clf = Ridge(
            parameter_keys="values",
        )
        clf.fit(X=X, y=y, alpha=alpha, sample_weight=sw)
        return clf

    num_properties = np.array([91, 100, 119, 221, 256])
    num_targets = np.array([1000, 2187])
    means = np.array([-0.5, 0, 0.1])
    regularizations = np.geomspace(1e-5, 1e5, 3)
    # For the tests using the paramaters above the number of properties always
    # needs to be less than the number of targets.
    # Otherwise, the property matrix will become singualar,
    # and the solution of the least squares problem would not be unique.

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    def test_ridge(self, num_properties, num_targets):
        """Test if ridge is working and all shapes are converted correctly.

        Test is performed for two blocks.
        """
        # Create input values
        X_arr = self.rng.random([2, num_targets, num_properties])
        y_arr = self.rng.random([2, num_targets, 1])
        alpha_arr = np.ones([2, 1, num_properties])
        sw_arr = np.ones([2, num_targets, 1])

        X = tensor_to_tensormap(X_arr)
        y = tensor_to_tensormap(y_arr)
        alpha = tensor_to_tensormap(alpha_arr)
        sw = tensor_to_tensormap(sw_arr)

        clf = Ridge(parameter_keys="values")
        clf.fit(X=X, y=y, alpha=alpha, sample_weight=sw)

        assert len(clf.weights) == 2
        assert clf.weights.block(0).values.shape[1] == num_properties
        assert clf.weights.block(1).values.shape[1] == num_properties

    def test_double_fit_call(self):
        """Test if regression works properly if fit method is called multiple times.

        This is especially important because we transform alpha into a dict durinb the
        fit."""
        num_targets = 10
        num_properties = 10
        num_blocks = 2

        X_arr = self.rng.random([num_blocks, num_targets, num_properties])
        y_arr = self.rng.random([num_blocks, num_targets, 1])
        alpha_arr = np.ones([num_blocks, 1, num_properties])

        X = tensor_to_tensormap(X_arr)
        y = tensor_to_tensormap(y_arr)
        alpha = tensor_to_tensormap(alpha_arr)

        clf = Ridge(parameter_keys="values")
        clf.fit(X=X, y=y, alpha=alpha)
        clf.fit(X=X, y=y, alpha=alpha)

        assert len(clf.weights) == num_blocks

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    @pytest.mark.parametrize("mean", means)
    def test_exact_no_regularization(self, num_properties, num_targets, mean):
        """Test that the weights predicted from the ridge regression
        solver match with the exact results (no regularization).
        """

        # Define properties and target properties
        X = self.rng.normal(mean, 1, size=(num_targets, num_properties))
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))
        y = X @ w_exact
        sample_w = np.ones((num_targets,))
        property_w = np.zeros((num_properties,))

        # Use solver to compute weights from X and y
        ridge_class = self.equisolve_solver_from_numpy_arrays(
            X, y, property_w, sample_w
        )
        w_solver = ridge_class.weights.block().values[0, :]

        # Check that the two approaches yield the same result
        assert_allclose(w_solver, w_exact, atol=1e-15, rtol=1e-10)

    @pytest.mark.parametrize("num_properties", [2, 4, 6])
    @pytest.mark.parametrize("num_targets", num_targets)
    @pytest.mark.parametrize("mean", means)
    def test_high_accuracy_ref_numpy_solver_regularization(
        self, num_properties, num_targets, mean
    ):
        """Test that the weights predicted from the ridge regression
        solver match with the exact results (regularization).
        As a benchmark, we use an explicit (purely numpy) solver of the
        regularized regression problem.
        We can only assume high accuracy for very small number of properties.
        Because the numerical inaccuracies vary too much between solvers.
        """

        # Define properties and target properties
        X = self.rng.normal(mean, 1, size=(num_targets, num_properties))
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))
        y = X @ w_exact
        sample_w = np.ones((num_targets,))
        property_w = np.zeros((num_properties,))

        # Use solver to compute weights from X and y
        ridge_class = self.equisolve_solver_from_numpy_arrays(
            X, y, property_w, sample_w
        )
        w_solver = ridge_class.weights.block().values[0, :]
        w_ref = numpy_solver(X, y, sample_w, property_w)

        # Check that the two approaches yield the same result
        assert_allclose(w_solver, w_ref, atol=1e-15, rtol=1e-10)

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    @pytest.mark.parametrize("mean", means)
    @pytest.mark.parametrize("regularization", regularizations)
    def test_approx_ref_numpy_solver_regularization_1(
        self, num_properties, num_targets, mean, regularization
    ):
        """Test that the weights predicted from the ridge regression
        solver match with the exact results (with regularization).
        As a benchmark, we use an explicit (purely numpy) solver of the
        regularized regression problem.
        """
        # Define properties and target properties
        X = self.rng.normal(mean, 1, size=(num_targets, num_properties))
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))
        # to obtain a low rank solution wrt. to number of properties

        y = X @ w_exact
        sample_w = np.ones((num_targets,))
        property_w = regularization * np.ones((num_properties,))

        # Use solver to compute weights from X and y
        ridge_class = self.equisolve_solver_from_numpy_arrays(
            X, y, property_w, sample_w
        )
        w_solver = ridge_class.weights.block().values[0, :]
        w_ref = numpy_solver(X, y, sample_w, property_w)

        # Check that the two approaches yield the same result
        assert_allclose(w_solver, w_ref, atol=1e-13, rtol=1e-8)

    num_properties = np.array([119, 512])
    num_targets = np.array([87, 511])
    means = np.array([-0.5, 0, 0.1])
    regularizations = np.geomspace(1e-5, 1e5, 3)
    # The tests using the paramaters above consider also the case where
    # num_features > num_target

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    @pytest.mark.parametrize("mean", means)
    @pytest.mark.parametrize("regularization", regularizations)
    def test_approx_ref_numpy_solver_regularization_2(
        self, num_properties, num_targets, mean, regularization
    ):
        """Test that the weights predicted from the ridge regression
        solver match with the exact results (with regularization).
        As a benchmark, we use an explicit (purely numpy) solver of the
        regularized regression problem.
        """
        # Define properties and target properties
        X = self.rng.normal(mean, 1, size=(num_targets, num_properties))
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))
        # to obtain a low rank solution wrt. to number of properties

        y = X @ w_exact
        sample_w = np.ones((num_targets,))
        property_w = regularization * np.ones((num_properties,))

        # Use solver to compute weights from X and y
        ridge_class = self.equisolve_solver_from_numpy_arrays(
            X, y, property_w, sample_w
        )
        w_solver = ridge_class.weights.block().values[0, :]
        w_ref = numpy_solver(X, y, sample_w, property_w)

        # Check that the two approaches yield the same result
        assert_allclose(w_solver, w_ref, atol=1e-13, rtol=1e-8)

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    @pytest.mark.parametrize("mean", means)
    def test_exact_low_rank_no_regularization(self, num_properties, num_targets, mean):
        """Test that the weights predicted from the ridge regression
        solver match with the exact results (no regularization).
        """
        # Define properties and target properties
        X = self.rng.normal(mean, 1, size=(num_targets, num_properties))
        # by reducing the rank to much smaller subset an exact solution can
        # still obtained of y, even if num_properties > num_targets
        low_rank = min(num_targets // 4, num_properties // 4)
        X[:, low_rank:] = 0
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))
        w_exact[low_rank:] = 0

        y = X @ w_exact
        sample_w = np.ones((num_targets,))
        property_w = np.zeros((num_properties,))

        # Use solver to compute weights from X and y
        ridge_class = self.equisolve_solver_from_numpy_arrays(
            X, y, property_w, sample_w
        )
        w_solver = ridge_class.weights.block().values[0, :]

        # Check that the two approaches yield the same result
        assert_allclose(w_solver, w_exact, atol=1e-15, rtol=1e-10)

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    @pytest.mark.parametrize("mean", means)
    def test_predict(self, num_properties, num_targets, mean):
        """Test that for given weights, the predicted target values on new
        data is correct."""
        # Define properties and target properties
        X = self.rng.normal(mean, 1, size=(num_targets, num_properties))
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))
        y = X @ w_exact
        sample_w = np.ones((num_targets,))
        property_w = np.zeros((num_properties,))

        # Use solver to compute weights from X and y
        ridge_class = self.equisolve_solver_from_numpy_arrays(
            X, y, property_w, sample_w
        )
        w_solver = ridge_class.weights.block().values[0, :]

        # Generate new data
        X_validation = self.rng.normal(mean, 1, size=(50, num_properties))
        y_validation_exact = X_validation @ w_solver
        y_validation_pred = ridge_class.predict(self.to_equistore(X_validation))

        # Check that the two approaches yield the same result
        assert_allclose(
            y_validation_pred.block().values[:, 0],
            y_validation_exact,
            atol=1e-15,
            rtol=1e-10,
        )

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    @pytest.mark.parametrize("mean", means)
    @pytest.mark.parametrize("regularization", np.array([1e40, 1e50]))
    def test_infinite_regularization(
        self, num_properties, num_targets, mean, regularization
    ):
        """Test that the weights predicted from the ridge regression
        solver match with the exact results (with regularization).
        As a benchmark, we use an explicit (purely numpy) solver of the
        regularized regression problem.
        """
        # Define properties and target properties
        X = self.rng.normal(mean, 1, size=(num_targets, num_properties))
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))
        y = X @ w_exact
        sample_w = np.ones((num_targets,))
        property_w = regularization * np.ones((num_properties,))

        # Use solver to compute weights from X and y
        ridge_class = self.equisolve_solver_from_numpy_arrays(
            X, y, property_w, sample_w
        )
        w_solver = ridge_class.weights.block().values[0, :]
        w_zeros = np.zeros((num_properties,))

        # Check that the two approaches yield the same result
        assert_allclose(w_solver, w_zeros, atol=1e-15, rtol=1e-10)

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    @pytest.mark.parametrize("mean", means)
    @pytest.mark.parametrize("scaling", np.array([1e-4, 1e3]))
    def test_consistent_weights_scaling(
        self, num_properties, num_targets, mean, scaling
    ):
        """Test of multiplying the weights same factor result in the same
        weights."""
        # Define properties and target properties
        X = self.rng.normal(mean, 1, size=(num_targets, num_properties))
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))
        y = X @ w_exact
        sample_w = np.ones((num_targets,))
        property_w = np.zeros((num_properties,))

        # Use solver to compute weights for the original and the scaled problems
        ridge_class = self.equisolve_solver_from_numpy_arrays(
            X, y, property_w, sample_w
        )
        w_ref = ridge_class.weights.block().values[0, :]
        ridge_class_scaled = self.equisolve_solver_from_numpy_arrays(
            X, y, scaling * property_w, scaling * sample_w
        )
        w_scaled = ridge_class_scaled.weights.block().values[0, :]

        # Check that the two approaches yield the same result
        assert_allclose(w_scaled, w_ref, atol=1e-13, rtol=1e-8)

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    @pytest.mark.parametrize("mean", means)
    @pytest.mark.parametrize("scaling", np.array([1e-4, 1e3]))
    def test_consistent_target_scaling(
        self, num_properties, num_targets, mean, scaling
    ):
        """Scaling the properties, the targets and the target weights by
        the same amount leads to the identical mathematical model."""
        # Define properties and target properties
        X = self.rng.normal(mean, 1, size=(num_targets, num_properties))
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))
        y = X @ w_exact
        sample_w = np.ones((num_targets,))
        property_w = np.zeros((num_properties,))

        # Use solver to compute weights for the original and the scaled problems
        ridge_class = self.equisolve_solver_from_numpy_arrays(
            X, y, property_w, sample_w
        )
        w_ref = ridge_class.weights.block().values[0, :]
        ridge_class_scaled = self.equisolve_solver_from_numpy_arrays(
            scaling * X, scaling * y, property_w, scaling * sample_w
        )
        w_scaled = ridge_class_scaled.weights.block().values[0, :]

        # Check that the two approaches yield the same result
        assert_allclose(w_scaled, w_ref, atol=1e-13, rtol=1e-8)

    def test_stability(self):
        """Test numerical stability of the solver."""
        # TODO
        pass

    def test_scalar_alpha(self):
        """Test ridge regression with scalar alpha."""
        # TODO
        pass

    def test_vector_alpha(self):
        """Test ridge regression with a different alpha per property."""
        # TODO
        pass

    def test_sample_weights(self):
        """Test ridge regression with a different value per target property."""
        # TODO
        pass

    def test_alpha_float(self):
        """Test float alpha"""
        X_arr = self.rng.random([1, 10, 10])
        y_arr = self.rng.random([1, 10, 1])
        alpha_arr = 2 * np.ones([1, 1, 10])

        X = tensor_to_tensormap(X_arr)
        y = tensor_to_tensormap(y_arr)
        alpha = tensor_to_tensormap(alpha_arr)

        clf = Ridge(parameter_keys="values")

        weights_arr = clf.fit(X=X, y=y, alpha=alpha).weights
        weights_float = clf.fit(X=X, y=y, alpha=2.0).weights

        assert_equal(weights_float.block().values, weights_arr.block().values)

    def test_alpha_wrong_type(self):
        """Test error raise if alpha is neither a float nor a TensorMap."""
        X_arr = self.rng.random([1, 10, 10])
        y_arr = self.rng.random([1, 10, 1])

        X = tensor_to_tensormap(X_arr)
        y = tensor_to_tensormap(y_arr)

        clf = Ridge(parameter_keys="values")

        with pytest.raises(ValueError, match="alpha must either be a float or"):
            clf.fit(X=X, y=y, alpha="foo")

    @pytest.mark.parametrize(
        "parameter_keys", [("values"), ("values", "positions"), ("positions")]
    )
    def test_parameter_keys(self, parameter_keys):
        """Test regression with different parameter_keys.

        Using a system with one atom per structure.
        """
        num_properties = 5
        num_targets = 10
        mean = 3.3

        X_arr = self.rng.normal(mean, 1, size=(4 * num_targets, num_properties))
        w_exact = self.rng.normal(mean, 3, size=(num_properties,))

        # Create training data
        X_values = X_arr[:num_targets]
        X_block = matrix_to_block(X_values)

        X_gradient_data = X_arr[num_targets:].reshape(num_targets, 3, num_properties)

        position_gradient_samples = Labels(
            ["sample", "structure", "atom"],
            np.array([[s, 1, 1] for s in range(num_targets)]),
        )

        X_block.add_gradient(
            parameter="positions",
            data=X_gradient_data,
            samples=position_gradient_samples,
            components=[Labels(["direction"], np.arange(3).reshape(-1, 1))],
        )

        X = TensorMap(Labels.single(), [X_block])

        # Create target data
        y_arr = X_arr @ w_exact

        y_values = y_arr[:num_targets].reshape(-1, 1)
        y_block = matrix_to_block(y_values)

        y_gradient_data = y_arr[num_targets:].reshape(num_targets, 3, 1)

        y_block.add_gradient(
            parameter="positions",
            data=y_gradient_data,
            samples=position_gradient_samples,
            components=[Labels(["direction"], np.arange(3).reshape(-1, 1))],
        )

        y = TensorMap(Labels.single(), [y_block])

        # Use no regularization.
        alpha = tensor_to_tensormap(
            np.zeros(num_properties).reshape(1, 1, -1), key_name="_"
        )

        clf = Ridge(parameter_keys=parameter_keys)
        clf.fit(X=X, y=y, alpha=alpha)

        assert_allclose(
            clf.weights.block().values, w_exact.reshape(1, -1), atol=1e-15, rtol=1e-10
        )

        # Test prediction
        X_pred = clf.predict(X)

        if "values" in parameter_keys:
            assert_allclose(X_pred.block().values, y.block().values)
        if "positions" in parameter_keys:
            assert_allclose(
                X_pred.block().gradient("positions").data,
                y.block().gradient("positions").data,
            )

    @pytest.mark.parametrize("components", [(True, False), (False, True)])
    def test_error_components(self, components):
        """Test error raise if X or y got components."""
        if components[0]:
            values_X = np.ones([1, 1, 1])
        else:
            values_X = np.ones([1, 1])

        if components[1]:
            values_y = np.ones([1, 1, 1])
        else:
            values_y = np.ones([1, 1])

        properties = Labels(["property"], np.arange(1).reshape(-1, 1))
        samples = Labels(["sample"], np.arange(1).reshape(-1, 1))

        X_block = TensorBlock(
            values=values_X,
            samples=samples,
            components=components[0] * [properties],
            properties=properties,
        )

        X = TensorMap(Labels.single(), [X_block])

        y_block = TensorBlock(
            values=values_y,
            samples=samples,
            components=components[1] * [properties],
            properties=properties,
        )
        y = TensorMap(Labels.single(), [y_block])

        clf = Ridge(parameter_keys="values")
        with pytest.raises(ValueError, match="contains components"):
            clf.fit(X=X, y=y)

    @pytest.mark.parametrize(
        "n_blocks", [[2, 1, 1, 1], [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]]
    )
    def test_error_different_blocks(self, n_blocks):
        """Test error raise if X, y, alpha or weights have different blocks."""

        X_arr = self.rng.random(
            [n_blocks[0], self.num_targets[0], self.num_properties[0]]
        )
        y_arr = self.rng.random([n_blocks[1], self.num_targets[0], 1])
        alpha_arr = np.ones([n_blocks[2], 1, self.num_properties[0]])
        sw_arr = np.ones([n_blocks[3], self.num_targets[0], 1])

        X = tensor_to_tensormap(X_arr)
        y = tensor_to_tensormap(y_arr)
        alpha = tensor_to_tensormap(alpha_arr)
        sw = tensor_to_tensormap(sw_arr)

        clf = Ridge(parameter_keys="values")
        with pytest.raises(ValueError, match="number of blocks"):
            clf.fit(X=X, y=y, alpha=alpha, sample_weight=sw)

    def test_error_properties(self):
        """Test error raise for non matching number of properties in X & alpha"""
        X, y, alpha = self.to_equistore(
            X_arr=np.ones([self.num_targets[0], self.num_properties[0]]),
            y_arr=np.ones(self.num_targets[0]),
            alpha_arr=np.ones(self.num_properties[0] + 1),
        )

        clf = Ridge(parameter_keys="values")
        with pytest.raises(ValueError, match="properties"):
            clf.fit(X=X, y=y, alpha=alpha)

    @pytest.mark.parametrize("extra_samples", [[0, 1], [1, 0]])
    def test_error_samples(self, extra_samples):
        """Test error raise for non matching number of samples in X & y"""

        X, y, alpha, sw = self.to_equistore(
            X_arr=np.ones([self.num_targets[0], self.num_properties[0]]),
            y_arr=np.ones(self.num_targets[0] + extra_samples[0]),
            alpha_arr=np.ones(self.num_properties[0]),
            sw_arr=np.ones(self.num_targets[0] + extra_samples[1]),
        )

        clf = Ridge(parameter_keys="values")
        with pytest.raises(ValueError, match="samples"):
            clf.fit(X=X, y=y, alpha=alpha, sample_weight=sw)

    @pytest.mark.parametrize("shapes", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
    def test_error_shape(self, shapes):
        """Test error raise if y, alpha & sw have more than 1 dimension."""

        X_arr = self.rng.random([1, self.num_targets[0], self.num_properties[0]])
        y_arr = self.rng.random([1, self.num_targets[0], shapes[0]])
        alpha_arr = np.ones([1, shapes[1], self.num_properties[0]])
        sw_arr = np.ones([1, self.num_targets[0], shapes[2]])

        X = tensor_to_tensormap(X_arr)
        y = tensor_to_tensormap(y_arr)
        alpha = tensor_to_tensormap(alpha_arr)
        sw = tensor_to_tensormap(sw_arr)

        clf = Ridge(parameter_keys="values")
        with pytest.raises(ValueError, match="Only one"):
            clf.fit(X=X, y=y, alpha=alpha, sample_weight=sw)

    def test_error_no_weights(self):
        """Test error raise if fit method was not called."""
        clf = Ridge(parameter_keys="values")
        with pytest.raises(ValueError, match="No weights"):
            clf.predict(1)
