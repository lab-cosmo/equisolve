# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import List, Set, Tuple, Union

from equistore import TensorMap
from equistore.operations import join, mean_over_samples, sum_over_samples

from rascaline import Composition, SoapRadialSpectrum, SoapPowerSpectrum

from .base import EquiScriptBase


class MultiSpectraScript(EquiScriptBase):
    def __init__(
        self,
        hypers: dict,
        spectra: Union[List[int], Set[int]] = None,
        *,
        feature_aggregation="mean",
        transformer_X=None,
        transformer_y=None,
        estimator=None
    ):
        super().__init__(
            hypers,
            feature_aggregation=feature_aggregation,
            transformer_X=transformer_X,
            transformer_y=transformer_y,
            estimator=estimator
        )
        self.spectra = spectra

    def _set_and_check_compute_parameters(self):
        if "SoapRadialSpectrum" not in self.hypers.keys():
            raise ValueError("No SoapRadialSpectrum given")
        if "SoapPowerSpectrum" not in self.hypers.keys():
            raise ValueError("No LodeSphericalExpansion given")

        valid_hyper_keys = set(["SoapRadialSpectrum", "SoapPowerSpectrum"])
        for key in self.hypers.keys():
            if key not in valid_hyper_keys:
                raise ValueError(
                    f"Only SoapRadialSpectrum and SoapPowerSpectrum as keys in hypers are allowed, but {key} found"
                )
        self._hypers = self.hypers

    def _set_and_check_fit_parameters(self) -> None:
        super()._set_and_check_fit_parameters()

        if self.spectra is None:
            self._spectra = set([0, 1, 2])
        else:
            # checks if only nu=0,1,2 are used by checking ∅ == spectra - {0,1,2}
            spectra_besides_nu012 = set(self.spectra).difference(set([0, 1, 2]))
            if len(spectra_besides_nu012) != 0:
                raise ValueError(
                    f"Only spectra 0, 1, 2 are supported but in addition {spectra_besides_nu012} were given"
                )
            self._spectra = self.spectra
        # TODO for now spectra is just ignored in compute and join

    def compute(self, **kwargs) -> Tuple[TensorMap, ...]:
        # input **kwargs same as for a rascaline calculator
        # outputs the (X0, X1, ..., XN), y TensorMap
        self._set_and_check_compute_parameters()

        descriptor_nu0 = Composition().compute(**kwargs)
        descriptor_nu1 = SoapRadialSpectrum(
            **self._hypers["SoapRadialSpectrum"]
        ).compute(**kwargs)
        descriptor_nu2 = SoapPowerSpectrum(**self._hypers["SoapPowerSpectrum"]).compute(
            **kwargs
        )

        return descriptor_nu0, descriptor_nu1, descriptor_nu2

    def _join(self, X: Tuple[TensorMap, ...]) -> TensorMap:
        # inputs the (X0, X1, ..., XN) TensorMaps
        # outputs the X, y TensorMaps where X joined X0, X1, ..., XN in a way defined here

        descriptor_nu0, descriptor_nu1, descriptor_nu2 = X

        # moving keys to properties
        descriptor_nu0 = descriptor_nu0.keys_to_properties(["species_center"])

        keys_to_move_to_samples = ["species_center"]
        keys_to_move_to_properties = ["species_neighbor"]
        descriptor_nu1 = descriptor_nu1.keys_to_samples(keys_to_move_to_samples)
        descriptor_nu1 = descriptor_nu1.keys_to_properties(keys_to_move_to_properties)

        keys_to_move_to_samples = ["species_center"]
        keys_to_move_to_properties = ["species_neighbor_1", "species_neighbor_2"]
        descriptor_nu2 = descriptor_nu2.keys_to_samples(keys_to_move_to_samples)
        descriptor_nu2 = descriptor_nu2.keys_to_properties(keys_to_move_to_properties)

        # aggregation
        samples_names = ["center", "species_center"]
        if self._feature_aggregation == "sum":
            descriptor_nu0 = sum_over_samples(descriptor_nu0, samples_names=["center"])
            samples_names = ["center", "species_center"]
            descriptor_nu1 = sum_over_samples(
                descriptor_nu1, samples_names=samples_names
            )
            descriptor_nu2 = sum_over_samples(
                descriptor_nu2, samples_names=samples_names
            )
        elif self._feature_aggregation == "mean":
            descriptor_nu0 = mean_over_samples(descriptor_nu0, samples_names=["center"])
            descriptor_nu1 = mean_over_samples(
                descriptor_nu1, samples_names=samples_names
            )
            descriptor_nu2 = mean_over_samples(
                descriptor_nu2, samples_names=samples_names
            )

        # joining
        X_nu01 = join([descriptor_nu0, descriptor_nu1], axis="properties")
        X_nu012 = join(
            [descriptor_nu0, descriptor_nu1, descriptor_nu2], axis="properties"
        )

        return X_nu012
