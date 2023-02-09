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

from equisolve.numpy.compose import TransformedTargetRegressor
from equisolve.numpy.models import Ridge
from equisolve.numpy.preprocessing import StandardScaler

from equisolve.numpy.utils import matrix_to_block, tensor_to_tensormap



class TestTransformedTargetRegressor:
    """Test the TransformedTargetRegressor."""

    rng = np.random.default_rng(0x122578748CFF12AD12)

    num_properties = np.array([91, 100, 119, 221, 256])
    num_targets = np.array([1000, 2187])

    @pytest.mark.parametrize("num_properties", num_properties)
    @pytest.mark.parametrize("num_targets", num_targets)
    def test_ttr(self, num_properties, num_targets):
        """Test if ridge is working and all shapes are converted correctly.

        Test is performed for two blocks.
        """
        # Create input values
        X_arr = self.rng.random([2, num_targets, num_properties])
        y_arr = self.rng.random([2, num_targets, 1])
        alpha_arr = np.ones([2, 1, num_properties])
        # TODO add as soon weights are supported
        sw_arr = np.ones([2, num_targets, 1])

        X = tensor_to_tensormap(X_arr)
        y = tensor_to_tensormap(y_arr)
        alpha = tensor_to_tensormap(alpha_arr)
        # TODO add as soon weights are supported
        #sw = tensor_to_tensormap(sw_arr)

        parameter_keys = "values"
        ridge = Ridge(parameter_keys=parameter_keys, alpha=alpha)
        standardizer = StandardScaler(parameter_keys=parameter_keys)
        clf = TransformedTargetRegressor(regressor=ridge, transformer=standardizer)
        clf.fit(X=X, y=y)
        clf.predict(X=X) # TODO put this in a separate test, right now its just testing if it runs through
        clf.score(X=X) # TODO put this in a separate test, right now its just testing if it runs through
