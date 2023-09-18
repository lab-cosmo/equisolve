from typing import List, Tuple, Union

import metatensor
import numpy as np
import pytest
from metatensor import Labels, TensorBlock, TensorMap

from equisolve.numpy.models import SorKernelRidge

from ..utilities import tensor_to_tensormap


def random_structured_tensor(
    n_blocks: int,
    n_structures: int,
    n_aggregate: Union[int, List[int]],
    n_properties: int,
    rng: np.random.Generator = None,
) -> TensorMap:
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(n_aggregate, list):
        raise NotImplementedError("list support for n_aggregate not implemented yet")

    n_samples = n_structures * n_aggregate
    X_arr = rng.random([n_blocks, n_samples, n_properties])
    X = tensor_to_tensormap(X_arr)

    sample_indices = np.repeat(np.arange(n_structures), n_samples // n_structures)
    structure_indices = sample_indices.copy()
    atom_indices = np.tile(np.arange(n_aggregate), n_samples // n_aggregate)
    samples_values = np.vstack((sample_indices, structure_indices, atom_indices)).T
    new_samples = Labels(
        names=["sample", "structure", "aggregate"], values=samples_values
    )
    new_blocks = []
    for block in X.blocks():
        # TODO support components
        assert len(block.components) == 0, "components not yet supported"
        new_block = TensorBlock(
            values=block.values,
            samples=new_samples,
            components=block.components,
            properties=block.properties,
        )
        # TODO create forces with finite difference, otherwise unlearnable
        #      I dont have a better idea
        for parameter, grad_block in block.gradients():
            new_block.add_gradient(parameter, grad_block)
        new_blocks.append(new_block)
    return TensorMap(X.keys, new_blocks)


def random_structured_target(
    tensor: TensorMap, rng: np.random.Generator = None
) -> Tuple[TensorMap, TensorMap]:
    """
    one cannot create just random properties, because there might be no linear
    relationship between input and target

    :param tensor:
        input tensor
    """
    # TODO function requires also a accumulate_key_names as SorKernelRidge
    if rng is None:
        rng = np.random.default_rng()

    weight_blocks = []
    for block in tensor.blocks():
        # TODO components are assumed to be zero in this construction
        #      add components
        assert len(block.components) == 0, "components not yet supported"
        weight_block = TensorBlock(
            values=rng.random([1, len(block.properties)]),
            samples=Labels.range("property", 1),
            components=[],
            properties=block.properties,
        )
        weight_blocks.append(weight_block)
    weight = TensorMap(tensor.keys, weight_blocks)
    target = metatensor.sum_over_samples(metatensor.dot(tensor, weight), "aggregate")
    return weight, target


class TestSorKernelRidge:
    """Test the class for a linear ridge regression."""

    @pytest.fixture(autouse=True)
    def set_random_generator(self):
        """Set the random generator to same seed before each test is run.
        Otherwise test behaviour is dependend on the order of the tests
        in this file and the number of parameters of the test.
        """
        self.rng = np.random.default_rng(0x1225787418FBDFD12)

    @pytest.mark.parametrize("alpha", np.geomspace(1e-5, 1e5, 3).tolist())
    @pytest.mark.parametrize("solver", ["solve", "lstsq", "RKHS", "RKHS-QR", "QR"])
    @pytest.mark.parametrize(
        "kernel",
        [
            {"type": "linear", "kwargs": {}},
            {"type": "polynomial", "kwargs": {"degree": 2}},
        ],
    )
    def test_fit_and_predict(self, alpha, solver, kernel):
        """Test if ridge is working and all shapes are converted correctly.
        Test is performed for two blocks.
        """
        num_structures = 5
        num_aggregates = 10  # PR COMMENT this is a more generic name for atoms
        # num_samples = num_structures * num_aggregates
        num_properties = 5
        num_pseudo_points = 5

        # Create input values
        y_arr = self.rng.random([2, num_structures, 1])

        X = random_structured_tensor(
            2, num_structures, num_aggregates, num_properties, self.rng
        )
        y = tensor_to_tensormap(y_arr)

        X_samples_values = np.vstack([block.samples.values for block in X.blocks()])
        X_samples_values = np.unique(X_samples_values, axis=0)
        selected_points = self.rng.choice(
            X_samples_values, num_pseudo_points, replace=False
        )
        pseudo_samples = Labels(names=X[0].samples.names, values=selected_points)
        X_pseudo = metatensor.slice(X, axis="samples", labels=pseudo_samples)

        sample_indices = np.arange(num_structures)
        structure_indices = sample_indices.copy()
        samples_values = np.vstack((sample_indices, structure_indices)).T
        new_samples = Labels(names=["sample", "structure"], values=samples_values)
        new_blocks = []
        for block in y.blocks():
            new_block = TensorBlock(
                values=block.values,
                samples=new_samples,
                components=block.components,
                properties=block.properties,
            )
            for parameter, grad_block in block.gradients():
                new_block.add_gradient(parameter, grad_block)
            new_blocks.append(new_block)
        y = TensorMap(y.keys, new_blocks)

        clf = SorKernelRidge()
        clf.fit(
            X=X,
            X_pseudo=X_pseudo,
            y=y,
            alpha=alpha,
            solver=solver,
            kernel_type=kernel["type"],
            kernel_kwargs=kernel["kwargs"],
        )
        assert len(clf.weights) == 2
        assert clf.weights.block(0).values.shape[1] == num_pseudo_points
        assert clf.weights.block(1).values.shape[1] == num_pseudo_points

        y_pred = clf.predict(X)

        for key in y.keys:
            assert y_pred.block(key).values.shape == y.block(key).values.shape
            assert y_pred.block(key).samples == y.block(key).samples
            assert y_pred.block(key).components == y.block(key).components
            assert y_pred.block(key).properties == y.block(key).properties

    @pytest.mark.parametrize(
        "args",
        [
            (1e-9, "lstsq", 1e-12, 1e-12),
            (1e-9, "RKHS", 1e-6, 1e-6),  # RKHS is a bit less accurate
            (1e-9, "RKHS-QR", 1e-12, 1e-12),
            (0.0, "lstsq", 1e-2, 1e-2),
            (0.0, "RKHS", 1e-2, 1e-2),
            (0.0, "RKHS-QR", 1e-2, 1e-2),
        ],
    )  # RKHS and RKHS-QR are less accurate for zero alpha
    def test_exact(self, args):
        """Test if ridge is working and all shapes are converted correctly.
        Test is performed for two blocks.
        """
        alpha, solver, atol, rtol = args

        num_structures = 5
        num_aggregates = 10
        num_samples = num_structures * num_aggregates
        num_properties = 3
        num_pseudo_points = num_samples

        # Create input values
        X = random_structured_tensor(
            2, num_structures, num_aggregates, num_properties, self.rng
        )
        _, y = random_structured_target(X, self.rng)

        X_samples_values = np.vstack([block.samples.values for block in X.blocks()])
        X_samples_values = np.unique(X_samples_values, axis=0)
        selected_points = self.rng.choice(
            X_samples_values, num_pseudo_points, replace=False
        )
        pseudo_samples = Labels(names=X[0].samples.names, values=selected_points)
        X_pseudo = metatensor.slice(X, axis="samples", labels=pseudo_samples)

        sample_indices = np.arange(num_structures)
        structure_indices = sample_indices.copy()
        samples_values = np.vstack((sample_indices, structure_indices)).T
        new_samples = Labels(names=["sample", "structure"], values=samples_values)
        new_blocks = []
        for block in y.blocks():
            new_block = TensorBlock(
                values=block.values,
                samples=new_samples,
                components=block.components,
                properties=block.properties,
            )
            for parameter, grad_block in block.gradients():
                new_block.add_gradient(parameter, grad_block)
            new_blocks.append(new_block)
        y = TensorMap(y.keys, new_blocks)

        clf = SorKernelRidge()
        clf.fit(X=X, X_pseudo=X_pseudo, y=y, alpha=alpha, solver=solver)
        assert len(clf.weights) == 2
        assert clf.weights.block(0).values.shape[1] == len(pseudo_samples)
        assert clf.weights.block(1).values.shape[1] == len(pseudo_samples)

        y_pred = clf.predict(X)

        # TODO remove print, but useful to see for now
        print(alpha, solver, np.mean(y[0].values / y_pred[0].values))
        metatensor.allclose_raise(y, y_pred, rtol=rtol, atol=atol)
