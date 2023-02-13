# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Set, Tuple

from equistore import TensorMap
from equistore.operations import join, mean_over_samples, sum_over_samples
from rascaline import LodeSphericalExpansion, SphericalExpansion

from .base import EquiScriptBase


class LodeKitScript(EquiScriptBase):
    def __init__(
        self,
        hypers,
        *,
        feature_aggregation="mean",
        transformer_X=None,
        transformer_y=None,
        estimator=None,
        parameter_keys=None,
    ):
        super().__init__(
            hypers,
            feature_aggregation=feature_aggregation,
            transformer_X=transformer_X,
            transformer_y=transformer_y,
            estimator=estimator,
            parameter_keys=parameter_keys,
        )

    def _set_and_check_fitting_parameters(self):
        if "SphericalExpansion" not in self.hypers.keys():
            raise ValueError("No SphericalExpansion given")
        if "LodeSphericalExpansion" not in self.hypers.keys():
            raise ValueError("No LodeSphericalExpansion given")
        valid_hyper_keys = set(["SphericalExpansion", "LodeSphericalExpansion"])
        for key in self.hypers.keys():
            if key not in valid_hyper_keys:
                raise ValueError(
                    f"Only SphericalExpansion and LodeSphericalExpansion as keys in hypers are allowed, but {key} found"
                )
        super()._set_and_check_fitting_parameters()

    def compute(self, **kwargs) -> Tuple[TensorMap, ...]:
        pass
        # input **kwargs same as for a rascaline calculator
        # outputs the (X0, X1, ..., XN), y TensorMaps

        # descriptor_co = Composition().compute(**kwargs)

        # descriptor_sr = SphericalExpansion(
        #     **self._hypers["SphericalExpansion"]
        # ).compute(**kwargs)
        # descriptor_ps_sr = compute_power_spectrum(descriptor_sr)

        # descriptor_lr = LodeSphericalExpansion(
        #     **self._hypers["LodeSphericalExpansion"]
        # ).compute(**kwargs)
        # descriptor_ps_lr = compute_power_spectrum(descriptor_sr, descriptor_lr)

        # return descriptor_co, descriptor_ps_sr, descriptor_ps_lr

    def join(self, X: Tuple[TensorMap, ...]) -> TensorMap:
        # inputs the (X0, X1, ..., XN) TensorMaps
        # outputs the X, y TensorMaps where X joined X0, X1, ..., XN in a way defined here

        pass

        # descriptor_co, descriptor_ps_sr, descriptor_ps_lr = X

        # # moving keys to properties
        # descriptor_co = descriptor_co.keys_to_properties(["species_center"])

        # keys_to_move_to_samples = ["species_center", "spherical_harmonics_l"]
        # keys_to_move_to_properties = ["species_neighbor_1", "species_neighbor_2"]

        # descriptor_ps_sr = descriptor_ps_sr.keys_to_samples(keys_to_move_to_samples)
        # descriptor_ps_sr = descriptor_ps_sr.keys_to_properties(
        #     keys_to_move_to_properties
        # )

        # descriptor_ps_lr = descriptor_ps_lr.keys_to_samples(keys_to_move_to_samples)
        # descriptor_ps_lr = descriptor_ps_lr.keys_to_properties(
        #     keys_to_move_to_properties
        # )

        # # aggregation
        # samples_names = ["center", "species_center", "spherical_harmonics_l"]
        # if self._feature_aggregation == "sum":
        #     descriptor_co = sum_over_samples(descriptor_co, samples_names=["center"])
        #     descriptor_ps_sr = sum_over_samples(
        #         descriptor_ps_sr, samples_names=samples_names
        #     )
        #     descriptor_ps_lr = sum_over_samples(
        #         descriptor_ps_lr, samples_names=samples_names
        #     )
        # elif self._feature_aggregation == "mean":
        #     descriptor_co = mean_over_samples(descriptor_co, samples_names=["center"])
        #     descriptor_ps_sr = mean_over_samples(
        #         descriptor_ps_sr, samples_names=samples_names
        #     )
        #     descriptor_ps_lr = mean_over_samples(
        #         descriptor_ps_lr, samples_names=samples_names
        #     )

        # # joining
        # X_sr = join([descriptor_co, descriptor_ps_sr], axis="properties")
        # X_lr = join(
        #     [descriptor_co, descriptor_ps_sr, descriptor_ps_lr], axis="properties"
        # )

        # return X_lr
