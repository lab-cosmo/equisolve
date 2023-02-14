# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from typing import List, Set, Dict, Tuple, Union

from equistore import TensorMap
from equistore.operations import join, mean_over_samples, sum_over_samples

from rascaline import Composition, SoapRadialSpectrum, SoapPowerSpectrum

from .base import EquiScriptBase

class MultiSpectraScript(EquiScriptBase):
    def __init__(
        self,
        hypers: dict,
        *,
        feature_aggregation="mean",
        transformer_X=None,
        transformer_y=None,
        estimator=None
    ):
        #TODO think about the order of hypers, need to be mention that join orders them
        super().__init__(
            hypers,
            feature_aggregation=feature_aggregation,
            transformer_X=transformer_X,
            transformer_y=transformer_y,
            estimator=estimator
        )

    def _set_and_check_compute_parameters(self):
        valid_hyper_keys = set(["Composition", "SoapRadialSpectrum", "SoapPowerSpectrum"])
        for key in self.hypers.keys():
            if key not in valid_hyper_keys:
                raise ValueError(
                    f"Only Composition, SoapRadialSpectrum and SoapPowerSpectrum as keys in hypers are allowed, but {key} found"
                )
        self._hypers = self.hypers

        # remove
        #if self.spectra_nu is None:
        #    self._spectra_nu = set([0, 1, 2])
        #else:
        #    # checks if only nu=0,1,2 are used by checking ∅ == spectra_nu - {0,1,2}
        #    spectra_nu_besides_nu012 = set(self.spectra_nu).difference(set([0, 1, 2]))
        #    if len(spectra_nu_besides_nu012) != 0:
        #        raise ValueError(
        #            f"Only spectra_nu 0, 1, 2 are supported but in addition {spectra_nu_besides_nu012} were given"
        #        )
        #    self._spectra_nu = self.spectra_nu

    def _set_and_check_fit_parameters(self) -> None:
        super()._set_and_check_fit_parameters()
        # TODO for now spectra is just ignored in compute and join

    # TODO switch dict to OrderedDict
    def compute(self, **kwargs) -> Dict[str, TensorMap]:
        # input **kwargs same as for a rascaline calculator
        # outputs the (X0, X1, ..., XN), y TensorMap
        self._set_and_check_compute_parameters()

        descriptors = {}
        if "Composition" in self._hypers.keys():
            descriptors["Composition"] = Composition().compute(**kwargs)
        if "SoapRadialSpectrum" in self._hypers:
            descriptors["SoapRadialSpectrum"] = SoapRadialSpectrum(
                **self._hypers["SoapRadialSpectrum"]
            ).compute(**kwargs)
        if "SoapPowerSpectrum" in self._hypers:
            descriptors["SoapPowerSpectrum"] = SoapPowerSpectrum(**self._hypers["SoapPowerSpectrum"]).compute(
                **kwargs
            )

        return descriptors

    def _move(self, X: Dict[str, TensorMap]) -> Dict[str, TensorMap]:
        # inputs the (X0, X1, ..., XN) TensorMaps
        # outputs the X, y TensorMaps where X joined X0, X1, ..., XN in a way defined here
        X_moved = {}
        if "Composition" in X.keys():
            keys_to_move_to_samples = ["species_center"]
            X_moved["Composition"] = X["Composition"].keys_to_properties(keys_to_move_to_samples)

        if "SoapRadialSpectrum" in X.keys():
            keys_to_move_to_samples = ["species_center"]
            keys_to_move_to_properties = ["species_neighbor"]
            X_moved["SoapRadialSpectrum"] = X["SoapRadialSpectrum"].keys_to_samples(keys_to_move_to_samples)
            X_moved["SoapRadialSpectrum"] = X_moved["SoapRadialSpectrum"].keys_to_properties(keys_to_move_to_properties)

        if "SoapPowerSpectrum" in X.keys():
            keys_to_move_to_samples = ["species_center"]
            keys_to_move_to_properties = ["species_neighbor_1", "species_neighbor_2"]
            X_moved["SoapPowerSpectrum"] = X["SoapPowerSpectrum"].keys_to_samples(keys_to_move_to_samples)
            X_moved["SoapPowerSpectrum"] = X_moved["SoapPowerSpectrum"].keys_to_properties(keys_to_move_to_properties)
        return X_moved

    def _aggregate(self, X: Dict[str, TensorMap]) -> Dict[str, TensorMap]:
        if self._feature_aggregation == "sum":
            aggregation_function  = sum_over_samples
        elif self._feature_aggregation == "mean":
            aggregation_function = mean_over_samples
        else:
            raise ValueError("Invalid aggregation_function.") # TODO make error message nice

        if "Composition" in X.keys():
            samples_names = ["center"]
            X["Composition"] = aggregation_function(X["Composition"], samples_names=["center"])
        if "SoapRadialSpectrum" in X.keys():
            samples_names = ["center", "species_center"]
            X["SoapRadialSpectrum"] = aggregation_function(
                X["SoapRadialSpectrum"], samples_names=samples_names
            )
        if "SoapPowerSpectrum" in X.keys():
            samples_names = ["center", "species_center"]
            X["SoapPowerSpectrum"] = aggregation_function(
                X["SoapPowerSpectrum"], samples_names=samples_names
            )
        return X

    def _join(self, X: Dict[str, TensorMap]) -> TensorMap:
        # joining
        # we do this to join them always in the same order
        return join(list(X.values()), axis="properties")
