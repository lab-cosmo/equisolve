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

        # Use block() dunction to check that TensorMap only has one block.
        block = property_tm.block()

        assert_equal(block.values, np.array(energies).reshape(-1, 1))
        assert_equal(
            block.gradient("positions").data,
            -np.concatenate(forces, axis=0).reshape(-1, 3, 1),
        )
        assert_equal(
            block.gradient("cell").data, -np.array(stress).reshape(-1, 3, 3, 1)
        )

    def test_properties_to_tensormap(self, energies, forces, stress):
        property_tm = properties_to_tensormap(energies, forces, stress)
        block = property_tm.block()

        assert_equal(block.values, np.array(energies).reshape(-1, 1))
        assert_equal(
            block.gradient("positions").data,
            np.concatenate(forces, axis=0).reshape(-1, 3, 1),
        )
        assert_equal(block.gradient("cell").data, np.array(stress).reshape(-1, 3, 3, 1))
