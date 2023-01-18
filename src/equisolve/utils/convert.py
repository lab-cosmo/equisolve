"""Functions for converting values into class:``equistore.TensorMap``s."""

from typing import List

import ase
import numpy as np
from equistore import Labels, TensorMap
from equistore.block import TensorBlock


def ase_to_tensormap(
    frames: List[ase.Atoms], energy: str = None, forces: str = None, stress: str = None
):
    """Store informations from :class:``ase.Atoms`` in a class:``equistore.TensorMap``.

    :param frames: ase.Atoms or list of ase.Atoms
    :param energy: key for extracting energy per structure
    :param forces: key for extracting atomic forces
    :param stress: key for extracting stress per structure
    """
    if not isinstance(frames, list):
        frames = [frames]

    values = [f.info[energy] for f in frames]

    if forces is not None:
        positions_gradients = [-f.arrays[forces] for f in frames]
    else:
        positions_gradients = None

    if stress is not None:
        cell_gradients = [-f.arrays[stress] for f in frames]
    else:
        cell_gradients = None

    return properties_to_tensormap(values, positions_gradients, cell_gradients)


def properties_to_tensormap(
    values: List[float],
    positions_gradients: List[np.ndarray] = None,
    cell_gradients: List[np.ndarray] = None,
    is_structure_property: bool = True,
):
    """Create a class:``equistore.TensorMap`` from array like properties.

    :values: List of length N
    :positions_gradients: list of length N
    :param cell_gradients: array of shape (N, 3, 3)
    """

    if not (is_structure_property):
        raise NotImplementedError(
            "Support for environment properties has not been implemented yet."
        )

    n_properties = len(values)

    samples = Labels(["structure"], np.arange(n_properties).reshape(-1, 1))

    block = TensorBlock(
        values=np.asarray(values).reshape(-1, 1),
        samples=samples,
        components=[],
        properties=Labels(["property"], np.array([(0,)])),
    )

    if positions_gradients is not None:
        if n_properties != len(positions_gradients):
            raise ValueError(
                f"given {n_properties} values but "
                f"{len(positions_gradients)} positions_gradients values"
            )

        data = np.concatenate(positions_gradients, axis=0)

        if data.shape[1] != 3:
            raise ValueError(
                f"positions_gradient must have 3 dimensions but has {data.shape[1]}"
            )

        position_gradient_samples = Labels(
            ["sample", "structure", "atom"],
            np.array(
                [
                    [s + a, s, a]
                    for s in range(len(positions_gradients))
                    for a in range(len(positions_gradients[s]))
                ]
            ),
        )

        block.add_gradient(
            parameter="positions",
            data=data.reshape(-1, 3, 1),
            samples=position_gradient_samples,
            components=[Labels(["direction"], np.arange(3).reshape(-1, 1))],
        )

    if cell_gradients is not None:
        if n_properties != len(cell_gradients):
            raise ValueError(
                f"given {n_properties} values but "
                f"{len(cell_gradients)} cell_gradients values"
            )

        data = np.asarray(cell_gradients)

        if data.shape[1:] != (3, 3):
            raise ValueError(
                "cell_gradient data must be a 3 x 3 matrix"
                f"but is {data.shape[1]} x {data.shape[2]}"
            )

        cell_gradient_samples = Labels(
            ["sample"], np.arange(n_properties).reshape(-1, 1)
        )

        components = [
            Labels(["direction_1"], np.arange(3).reshape(-1, 1)),
            Labels(["direction_2"], np.arange(3).reshape(-1, 1)),
        ]

        block.add_gradient(
            parameter="cell",
            data=data.reshape(-1, 3, 3, 1),
            samples=cell_gradient_samples,
            components=components,
        )

    return TensorMap(Labels.single(), [block])
