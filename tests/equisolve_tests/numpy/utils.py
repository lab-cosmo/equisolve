# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from metatensor import Labels, TensorBlock
from numpy.testing import assert_equal

from equisolve.numpy.utils import array_from_block


class Testarray_from_block:
    """Test the conversion of a TensorBlock into a numpy array."""

    n_samples = 3
    n_properties = 2

    @pytest.fixture
    def values(self):
        return np.arange(self.n_samples * self.n_properties).reshape(
            self.n_samples, self.n_properties
        )

    @pytest.fixture
    def gradient_values(self):
        return np.arange(self.n_samples * self.n_properties * 3).reshape(
            self.n_samples, 3, self.n_properties
        )

    @pytest.fixture
    def block(self, values):
        properties = Labels(["property"], np.arange(self.n_properties).reshape(-1, 1))
        samples = Labels(["sample"], np.arange(self.n_samples).reshape(-1, 1))

        block = TensorBlock(
            values=values, samples=samples, components=[], properties=properties
        )
        return block

    @pytest.fixture
    def block_gradient(self, block, gradient_values):
        gradient = TensorBlock(
            values=gradient_values,
            samples=block.samples,
            components=[Labels(["direction"], np.arange(3).reshape(-1, 1))],
            properties=block.properties,
        )

        block.add_gradient("positions", gradient)
        return block

    def test_values(self, block, values):
        """Test extraction of values"""
        block_mat = array_from_block(block)

        assert_equal(block_mat, values)

    def test_values_gradients(self, block_gradient, values, gradient_values):
        block_mat = array_from_block(block_gradient)

        assert_equal(
            block_mat,
            np.vstack(
                [values, gradient_values.reshape(self.n_samples * 3, self.n_properties)]
            ),
        )

    def test_components(self, values):
        properties = Labels(["property"], np.arange(self.n_properties).reshape(-1, 1))
        samples = Labels(["sample"], np.arange(self.n_samples).reshape(-1, 1))

        values = np.arange(self.n_samples * self.n_properties**2).reshape(
            self.n_samples, self.n_properties, self.n_properties
        )

        block = TensorBlock(
            values=values,
            samples=samples,
            components=[properties],
            properties=properties,
        )

        block_mat = array_from_block(block)

        assert_equal(
            block_mat,
            values.reshape(self.n_samples * self.n_properties, self.n_properties),
        )
