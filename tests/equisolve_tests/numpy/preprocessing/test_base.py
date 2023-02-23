# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose

from equisolve.numpy.preprocessing import StandardScaler

from ..utils import random_single_block_no_components_tensor_map


class TestStandardScaler:
    """Test the StandardScaler."""

    rng = np.random.default_rng(0x122578748CFF12AD12)
    n_strucs = 5

    X = random_single_block_no_components_tensor_map()

    absolute_tolerance = 1e-14
    relative_tolerance = 1e-14

    @pytest.mark.parametrize("with_mean", [True, False])
    @pytest.mark.parametrize("with_std", [True, False])
    def test_standard_scaler_transform(self, with_mean, with_std):
        parameter_keys = ["values", "positions"]
        st = StandardScaler(
            parameter_keys=parameter_keys,
            with_mean=with_mean,
            with_std=with_std,
            column_wise=False,
        ).fit(self.X)
        X_t = st.transform(self.X)

        X_values = X_t.block().values
        if with_mean:
            assert_allclose(
                np.mean(X_values, axis=0),
                0,
                atol=self.absolute_tolerance,
                rtol=self.relative_tolerance,
            )
        if with_std:
            assert_allclose(
                np.sqrt(np.sum(np.var(X_values, axis=0))),
                1,
                atol=self.absolute_tolerance,
                rtol=self.relative_tolerance,
            )

        parameter_keys.remove("values")
        for parameter in parameter_keys:
            X_grad = X_t.block().gradient(parameter)
            X_grad = X_grad.data.reshape(-1, X_grad.data.shape[-1])
            if with_mean:
                assert_allclose(np.mean(X_grad, axis=0), 0, atol=1e-14, rtol=1e-14)
            if with_std:
                assert_allclose(
                    np.sqrt(np.sum(np.var(X_grad, axis=0))),
                    1,
                    atol=self.absolute_tolerance,
                    rtol=self.relative_tolerance,
                )

    @pytest.mark.parametrize("with_mean", [True, False])
    @pytest.mark.parametrize("with_std", [True, False])
    def test_standard_scaler_inverse_transform(self, with_mean, with_std):
        parameter_keys = ["values", "positions"]
        st = StandardScaler(
            parameter_keys=parameter_keys,
            with_mean=with_mean,
            with_std=with_std,
            column_wise=False,
        ).fit(self.X)
        X_t_inv_t = st.inverse_transform(st.transform(self.X))

        assert_allclose(
            self.X.block().values,
            X_t_inv_t.block().values,
            atol=self.absolute_tolerance,
            rtol=self.relative_tolerance,
        )

        parameter_keys.remove("values")
        for parameter in parameter_keys:
            X_grad = self.X.block().gradient(parameter)
            assert_allclose(
                X_grad.data,
                X_t_inv_t.block().gradient(parameter).data,
                atol=self.absolute_tolerance,
                rtol=self.relative_tolerance,
            )
