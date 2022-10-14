# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

from ..utils import normalize, structure_sum


class LinearModel:
    """Linear Model"""

    def __init__(self, normalize, regularizer):
        self.regularizer = regularizer
        self.normalize = normalize

        self.weights = None
        self.baseline = None

    def fit(self, ps, energies, forces=None):
        ps_per_structure = structure_sum(ps)

        if self.normalize:
            ps_per_structure = normalize(ps_per_structure)

        block = ps_per_structure.block()
        assert len(block.components) == 0

        X = block.values

        self.baseline = energies.mean()
        Y = energies.reshape(-1, 1) - self.baseline

        delta = energies.std()
        structures = np.unique(block.samples["structure"])
        n_atoms_per_structure = []
        for structure in structures:
            n_atoms = np.sum(block.samples["structure"] == structure)
            n_atoms_per_structure.append(float(n_atoms))

        energy_regularizer = (
            np.sqrt(np.array_like(energies, n_atoms_per_structure))
            * self.regularizer[0]
            / delta
        )

        if forces is not None:
            gradient = block.gradient("positions")
            X_grad = gradient.data.reshape(3 * len(gradient.samples), X.shape[1])

            energy_grad = -forces.reshape(X_grad.shape[0], 1)

            # TODO: this assume the atoms are in the same order in X_grad &
            # forces
            Y = np.vstack([Y, energy_grad])
            X = np.vstack([X, X_grad])

        # solve weights as `w = X.T (X X.T + λ I)^{-1} Y` instead of the usual
        # `w = (X.T X + λ I)^{-1} X.T Y` since that allow us to regularize
        # energy & forces separately.
        #
        # cf https://stats.stackexchange.com/a/134068
        X_XT = X @ X.T

        property_idx = np.arange(len(energies))
        X_XT[property_idx, property_idx] += energy_regularizer

        if forces is not None:
            forces_regularizer = self.regularizer[1] / delta
            grad_idx = np.arange(len(energies), stop=X_XT.shape[0])
            X_XT[grad_idx, grad_idx] += forces_regularizer

        # TODO: can this use linalg_solve instead?
        self.weights = X.T @ np.linalg.inv(X_XT) @ Y

    def predict(self, ps, with_forces=False):
        if self.weights is None:
            raise Exception("call fit first")

        ps_per_structure = structure_sum(ps)

        if self.normalize:
            ps_per_structure = normalize(ps_per_structure)

        block = ps_per_structure.block()
        assert len(block.components) == 0

        X = block.values

        energies = X @ self.weights + self.baseline

        if with_forces:
            gradient = block.gradient("positions")
            X_grad = gradient.data.reshape(-1, 3, self.weights.shape[0])

            forces = -X_grad @ self.weights
            forces = forces.reshape(-1, 3)
        else:
            forces = None

        return energies, forces
