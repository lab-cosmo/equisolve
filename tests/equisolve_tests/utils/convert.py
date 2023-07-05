# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

import ase
import numpy as np
import pytest
from numpy.testing import assert_equal

from equisolve.utils import ase_to_tensormap, properties_to_tensormap


class TestConvert:
    """Test the converting functions.

    The functions `energies`, `forces` and `stress` create lists of
    numpy arrays.
    """

    rng = np.random.default_rng(0x122578748CFF12AD12)
    n_strucs = 2
    n_atoms = 4

    @pytest.fixture
    def energies(self):
        return [i for i in self.rng.random(self.n_strucs)]

    @pytest.fixture
    def forces(self):
        return [i for i in self.rng.random([self.n_strucs, self.n_atoms, 3])]

    @pytest.fixture
    def stress(self):
        return [i for i in self.rng.random([self.n_strucs, 3, 3])]

    def test_ase_to_tensormap(self, energies, forces, stress):
        frames = []
        for i in range(len(energies)):
            frame = ase.Atoms(self.n_atoms * "H")
            frame.info["energy"] = energies[i]
            frame.arrays["forces"] = forces[i]
            frame.arrays["stress"] = stress[i]
            frames.append(frame)

        property_tm = ase_to_tensormap(frames, "energy", "forces", "stress")

        # Use `[0]` function without parameters to check that TensorMap
        # only has one block.
        block = property_tm[0]

        assert_equal(block.values, np.array(energies).reshape(-1, 1))
        assert_equal(
            block.gradient("positions").values,
            -np.concatenate(forces, axis=0).reshape(-1, 3, 1),
        )
        assert_equal(
            block.gradient("cell").values, -np.array(stress).reshape(-1, 3, 3, 1)
        )

    def test_properties_to_tensormap(self, energies, forces, stress):
        property_tm = properties_to_tensormap(energies, forces, stress)
        block = property_tm[0]

        assert_equal(block.values, np.array(energies).reshape(-1, 1))
        assert_equal(
            block.gradient("positions").values,
            np.concatenate(forces, axis=0).reshape(-1, 3, 1),
        )
        assert_equal(
            block.gradient("cell").values, np.array(stress).reshape(-1, 3, 3, 1)
        )

    def test_position_gradient_samples(self, energies, forces):
        """Test that the position gradients sample labels agree with the convention."""

        property_tm = properties_to_tensormap(energies, forces)
        block = property_tm[0]

        samples = block.gradient("positions").samples

        assert samples.names == [
            "sample",
            "structure",
            "atom",
        ]

        assert_equal(
            samples.values.tolist(),
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 1, 3],
            ],
        )

    def test_cell_gradient_samples(self, energies, stress):
        """Test that the cell gradients sample labels agree with the convention."""

        property_tm = properties_to_tensormap(energies, cell_gradients=stress)
        block = property_tm[0]

        samples = block.gradient("cell").samples

        assert samples.names == ["sample"]

        assert_equal(samples.values.tolist(), [[0], [1]])

    def test_position_gradient_wrong_n_samples(self, energies):
        """Test error raise if the length values and positions_gradients not equal."""
        forces = [i for i in self.rng.random([self.n_strucs + 1, self.n_atoms, 3])]

        with pytest.raises(ValueError, match="positions_gradients values"):
            properties_to_tensormap(energies, positions_gradients=forces)

    def test_position_gradient_wrong_shape(self, energies):
        """Test error raise if position_gradients do not have 3 columns."""
        forces = [i for i in self.rng.random([self.n_strucs, self.n_atoms, 2])]

        with pytest.raises(ValueError, match="must have 3 columns but have 2"):
            properties_to_tensormap(energies, positions_gradients=forces)

    def test_cell_gradient_wrong_n_samples(self, energies):
        """Test error raise if the length values and cell_gradients not equal."""
        stress = [i for i in self.rng.random([self.n_strucs + 1, 2, 3])]
        with pytest.raises(ValueError, match="cell_gradients values"):
            properties_to_tensormap(energies, cell_gradients=stress)

    def test_cell_gradient_wrong_shape(self, energies):
        """Test error raise if cell_gradients are not 3x3 matrices."""

        stress = [i for i in self.rng.random([self.n_strucs, 2, 3])]

        with pytest.raises(ValueError, match="data must be a 3 x 3 matrix"):
            properties_to_tensormap(energies, cell_gradients=stress)
