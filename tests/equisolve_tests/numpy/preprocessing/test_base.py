# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

import os

import ase.io
import numpy as np
import pytest
from numpy.testing import assert_allclose
from rascaline import SoapPowerSpectrum

from equisolve.numpy.preprocessing import StandardScaler


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


class TestStandardScaler:
    """Test the StandardScaler."""

    rng = np.random.default_rng(0x122578748CFF12AD12)
    n_strucs = 5
    frames = ase.io.read(os.path.join(ROOT, "./examples/dataset.xyz"), f":{n_strucs}")

    hypers = {
        "cutoff": 5.0,
        "max_radial": 2,
        "max_angular": 2,
        "atomic_gaussian_width": 0.3,
        "center_atom_weight": 1.0,
        "radial_basis": {
            "Gto": {},
        },
        "cutoff_function": {
            "ShiftedCosine": {"width": 0.5},
        },
    }

    calculator = SoapPowerSpectrum(**hypers)

    descriptor = calculator.compute(frames, gradients=["positions"])
    # Move all keys into properties
    descriptor = descriptor.keys_to_properties(
        ["species_center", "species_neighbor_1", "species_neighbor_2"]
    )
    # ``X`` contains a represenantion with respect to central atoms. However
    # our energies as target data is per structure. Therefore, we convert the
    # central atom representation into a structure wise represention by summing all
    # properties per structure.
    from equistore.operations import sum_over_samples

    X = sum_over_samples(descriptor, ["center"])

    @pytest.fixture
    def energies(self):
        return [i for i in self.rng.random(self.n_strucs)]

    def test_standard_scaler_transform(self):
        st = StandardScaler(["values", "positions"]).fit(self.X)
        X_t = st.transform(self.X)

        X_values = X_t.block().values
        assert_allclose(np.mean(X_values, axis=0), 0, atol=1e-14, rtol=1e-14)
        assert_allclose(
            np.sqrt(np.sum(np.var(X_values, axis=0))), 1, atol=1e-14, rtol=1e-14
        )

        for _, X_grad in X_t.block().gradients():
            X_grad = X_grad.data.reshape(-1, X_grad.data.shape[-1])
            assert_allclose(np.mean(X_grad, axis=0), 0, atol=1e-14, rtol=1e-14)
            assert_allclose(
                np.sqrt(np.sum(np.var(X_grad, axis=0))), 1, atol=1e-14, rtol=1e-14
            )

    def test_standard_scaler_inverse_transform(self):
        st = StandardScaler(["values", "positions"]).fit(self.X)
        X_t_inv_t = st.inverse_transform(st.transform(self.X))

        assert_allclose(
            self.X.block().values, X_t_inv_t.block().values, atol=1e-14, rtol=1e-14
        )

        for parameter, X_grad in self.X.block().gradients():
            assert_allclose(
                X_grad.data,
                X_t_inv_t.block().gradient(parameter).data,
                atol=1e-14,
                rtol=1e-14,
            )
