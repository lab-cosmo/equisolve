from abc import ABCMeta, abstractmethod
from copy import deepcopy

from typing import List, Set, Optional, Union, Tuple, TypeVar

from equisolve.numpy.models.linear_model import Ridge

from rascaline import Composition, LodeSphericalExpansion, SphericalExpansion

from rascaline import Composition, SoapRadialSpectrum, SoapPowerSpectrum

from equistore import TensorBlock, TensorMap
from equistore.operations import join, ones_like, slice, sum_over_samples, mean_over_samples

# TODO from utils.models.soap import compute_power_spectrum

import ase

import numpy as np

# Workaround for typing Self with inheritance for python <3.11
# see https://peps.python.org/pep-0673/
TEquiKitScript= TypeVar("TEquiKitScript", bound="EquiKitScript")

class EquiKitScript(metaclass=ABCMeta):
    """
    An EquiScript is a merge of a representation calculator and a ML model.

    EquiKitScript supports scikit-learn like transformers and estimators that
    have a fit and predict function and can handle equistore objects.
    For some applications (e.g. long-range interaction, multi-spectra) we want to apply transformers
    seperately on different representations (e.g. because they are on different scales)
    and join them together for the estimation.

    Workflow
    ```python
        hypers = ...
        script = EquiKitScript(hypers, parameter_keys=["values", "positions"])
        Xi = script.compute(frames) # output is (X0, ... XN)
        y = ase_to_tensormap(frames, energy="energy", forces="forces")

        script.fit(Xi, y, **estimator_fit_kwargs)
        # fit logic:
        #  transformer_X[0].transform(X0), ..., transformer_X[N].transform(XN) | transformer_y.transform(y)
        #               X0_t,              ...,              XN_t              |            y_t
        #
        #  join(X0_t, ..., XN_t)
        #           X_jt
        #
        #  estimator.fit(X_jt, y_t)

        script.forward(Xi)
    ```

    COMMENT class name is too general, can we distinguish it to Hamiltonian learning and CG-NN applications where postprocessing happens after the join to compute errors?
            Might be not important because these applications require gradient optimization, so they would be in any not within the Scikit scope

    COMMENT The logic is a bit unnecessary entangled with atomistic data, but I would not bother with it for now
    """
    def __init__(self, hypers, *, feature_aggregation="mean", transformer_X=None, transformer_y=None, estimator=None):
        self.hypers = hypers

        self.feature_aggregation = feature_aggregation

        if estimator is None:
            raise NotImplemented("Ridge still needs a good default alpha value, before we can support default parameter")
            #empty_tm = TensorMap(keys=Labels.single(), blocks=[TensorBlock(
            #    values=np.empty((0,1)),
            #    samples=Labels(names=["samples"], values=np.empty((0,1), dtype=np.int32)),
            #    components=[],
            #    properties=Labels.single(),
            #)])
            #self.estimator = Ridge(parameter_keys=parameter_keys, alpha=empty_tm)

    def fit(self, X: Tuple[TensorMap, ...], y: TensorMap, **kwargs) -> TEquiKitScript:
        # X : (X0, X1, ..., XN)
        self._set_and_check_fitting_parameters()
        if "transformer_X" not in kwargs.keys():
            kwargs["transformer_X"] = {}
        if "transformer_y" not in kwargs.keys():
            kwargs["transformer_y"] = {}
        if "estimator" not in kwargs.keys():
            kwargs["estimator"] = {}

        if self.transformer_X is None:
            self._transformers_X = None
        else:
            self._transformers_X = []
            for i in range(len(T)):
                self._transformers_X.append( deepcopy(transformer_X).fit(X[i], **kwargs["transformer_X"]) )

        if self.transformer_y is None:
            self._transformer_y = None
        else:
            self._transformer_y = deepcopy(self.transformer_y).fit(y, **kwargs["transformer_y"])

        X = tuple(self._transformers_X[i].transform(X[i]) for i in range(len(self._transformers_X)))
        y = self._transformer_y.transform(y)

        X = self.transform_join(X)

        self._estimator = deepcopy(self.estimator).fit(X, y, **kwargs["estimator"])

    def forward(self, X: Tuple[TensorMap, ...]) -> TensorMap:
        # TODO check if is fitted

        if self._transformers_X is not None:
            X = tuple(self._transformers_X[i].transform(X[i]) for i in range(len(self._transformers_X)))
        if self._transformer_y is not None:
            y = self._transformer_y.transform(y)

        X = self.join(X)

        y_pred = self._estimator.predict(X, y)
        if self._transformer_y is not None:
            y_pred = self._transformer_y.inverse_transform(y_pred)
        return y_pred

    def score(self, X: Tuple[TensorMap, ...], y) -> List[float]:
        # TODO(low-prio) add support for more error functions
        y_pred = self.predict(X, y)
        return rmse(y, y_pred, self._parameter_keys)

    def _set_and_check_fitting_parameters(self) -> None:
        # TODO check if parameter_keys are consistent over the transformers and estimators
        #      and if not throw warning

        if self.feature_aggregation not in ["sum", "meam"]:
            raise ValueError("Only 'sum' and 'mean' are supported for feature_aggregation")

        # TODO would rename to _fit_*
        # we save all member variables, to have the member variables used in the last fit
        self._hypers = self.hypers
        self._feature_aggregation = self._feature_aggregation
        self._parameter_keys = self.parameter_keys

    @abstractmethod
    def join(self, X: Tuple[TensorMap, ...]) -> TensorMap:
        # TODO make error message sound nicer, double check if Metaclass is basically useless and does not do this
        raise NotImplemented("join function not implemented")

    @abstractmethod
    def compute(self, **kwargs) -> Tuple[TensorMap, ...]:
        # TODO make error message sound nicer, double check if Metaclass is basically useless and does not do this
        raise NotImplemented("compute function not implemented")


class MultiSpectraKitScript(EquiKitScript):
    def __init__(self, hypers, spectra: Union[List[int], Set[int]] = None, *,
                 feature_aggregation="mean", transformer_X=None, transformer_y=None, estimator=None, parameter_keys=None):
        super().__init__(hypers,
                         feature_aggregation=feature_aggregation,
                         transformer_X=transformer_X,
                         transformer_y=transformer_y,
                         estimator=estimator,
                         parameter_keys=parameter_keys
                        )

    def _set_and_check_fitting_parameters(self):
        if "SoapRadialSpectrum" not in self.hypers.keys():
            raise ValueError("No SoapRadialSpectrum given")
        if "SoapPowerSpectrum" not in self.hypers.keys():
            raise ValueError("No LodeSphericalExpansion given")

        valid_hyper_keys = set(["SoapRadialSpectrum", "SoapPowerSpectrum"])
        for key in self.hypers.keys():
            if key not in valid_hyper_keys:
                raise ValueError(f"Only SoapRadialSpectrum and SoapPowerSpectrum as keys in hypers are allowed, but {key} found")
        super()._set_and_check_fitting_parameters()

        if self.spectra is None:
            self._spectra = set(0,1,2)
        else:
            # checks if only nu=0,1,2 are used by checking ∅ == spectra - {0,1,2}
            spectra_besides_nu012 = set(self.spectra).difference(set([0,1,2]))
            if len(spectra_besides_nu012) != 0:
                raise ValueError(f"Only spectra 0, 1, 2 are supported but in addition {spectra_besides_nu012} were given")
            self._spectra = self.spectra
        # TODO for now spectra is just ignored in compute and join

    def compute(self, **kwargs) -> Tuple[TensorMap, ...]:
        # input **kwargs same as for a rascaline calculator
        # outputs the (X0, X1, ..., XN), y TensorMap

        descriptor_nu0 = Composition().compute(**kwargs)
        descriptor_nu1 = SoapRadialSpectrum(**self._hypers["SoapRadialSpectrum"]).compute(**kwargs)
        descriptor_nu2 = SoapPowerSpectrum(**self._hypers["SoapPowerSpectrum"]).compute(**kwargs)

        return descriptor_nu0, descriptor_nu1, descriptor_nu2

    def join(self, X: Tuple[TensorMap, ...]) -> TensorMap:
        # inputs the (X0, X1, ..., XN) TensorMaps
        # outputs the X, y TensorMaps where X joined X0, X1, ..., XN in a way defined here

        descriptor_nu0, descriptor_nu1, descriptor_nu2 = X

        # moving keys to properties
        descriptor_nu0 = descriptor_nu0.keys_to_properties(["species_center"])

        keys_to_move_to_samples = ["species_center", "spherical_harmonics_l"]
        keys_to_move_to_properties = ["species_neighbor_1", "species_neighbor_2"]

        descriptor_nu1  = descriptor_nu1.keys_to_samples(keys_to_move_to_samples)
        descriptor_nu1 = descriptor_nu1.keys_to_properties(keys_to_move_to_properties)

        descriptor_nu2  = descriptor_nu2.keys_to_samples(keys_to_move_to_samples)
        descriptor_nu2 = descriptor_nu2.keys_to_properties(keys_to_move_to_properties)

        # aggregation
        samples_names = ['center', 'species_center', 'spherical_harmonics_l']
        if self._feature_aggregation == 'sum':
            descriptor_nu0 = sum_over_samples(descriptor_nu0, samples_names=['center'])
            descriptor_nu1 = sum_over_samples(descriptor_nu1, samples_names=samples_names)
            descriptor_nu2 = sum_over_samples(descriptor_nu2, samples_names=samples_names)
        elif self._feature_aggregation == 'mean':
            descriptor_nu0 = mean_over_samples(descriptor_nu0, samples_names=['center'])
            descriptor_nu1 = mean_over_samples(descriptor_nu1, samples_names=samples_names)
            descriptor_nu2 = mean_over_samples(descriptor_nu2, samples_names=samples_names)

        # joining
        X_nu01 = join([descriptor_nu0, descriptor_nu1], axis="properties")
        X_nu012 = join([descriptor_nu0, descriptor_nu1, descriptor_nu2], axis="properties")

        return X_nu012



class LodeKitScript(EquiKitScript):
    def __init__(self, hypers, *,
                 feature_aggregation="mean", transformer_X=None, transformer_y=None, estimator=None, parameter_keys=None):
        super().__init__(hypers,
                         feature_aggregation=feature_aggregation,
                         transformer_X=transformer_X,
                         transformer_y=transformer_y,
                         estimator=estimator,
                         parameter_keys=parameter_keys
                        )


    def _set_and_check_fitting_parameters(self):
        if "SphericalExpansion" not in self.hypers.keys():
            raise ValueError("No SphericalExpansion given")
        if "LodeSphericalExpansion" not in self.hypers.keys():
            raise ValueError("No LodeSphericalExpansion given")
        valid_hyper_keys = set(["SphericalExpansion", "LodeSphericalExpansion"])
        for key in self.hypers.keys():
            if key not in valid_hyper_keys:
                raise ValueError(f"Only SphericalExpansion and LodeSphericalExpansion as keys in hypers are allowed, but {key} found")
        super()._set_and_check_fitting_parameters()


    def compute(self, **kwargs) -> Tuple[TensorMap, ...]:
        # input **kwargs same as for a rascaline calculator
        # outputs the (X0, X1, ..., XN), y TensorMaps

        descriptor_co = Composition().compute(**kwargs)

        descriptor_sr = SphericalExpansion(**self._hypers["SphericalExpansion"]).compute(**kwargs)
        descriptor_ps_sr = compute_power_spectrum(descriptor_sr)

        descriptor_lr = LodeSphericalExpansion(**self._hypers["LodeSphericalExpansion"]).compute(**kwargs)
        descriptor_ps_lr = compute_power_spectrum(descriptor_sr, descriptor_lr)

        return descriptor_co, descriptor_ps_sr, descriptor_ps_lr

    def join(self, X: Tuple[TensorMap, ...]) -> TensorMap:
        # inputs the (X0, X1, ..., XN) TensorMaps
        # outputs the X, y TensorMaps where X joined X0, X1, ..., XN in a way defined here

        descriptor_co, descriptor_ps_sr, descriptor_ps_lr = X

        # moving keys to properties
        descriptor_co = descriptor_co.keys_to_properties(["species_center"])

        keys_to_move_to_samples = ["species_center", "spherical_harmonics_l"]
        keys_to_move_to_properties = ["species_neighbor_1", "species_neighbor_2"]

        descriptor_ps_sr = descriptor_ps_sr.keys_to_samples(keys_to_move_to_samples)
        descriptor_ps_sr = descriptor_ps_sr.keys_to_properties(keys_to_move_to_properties)

        descriptor_ps_lr = descriptor_ps_lr.keys_to_samples(keys_to_move_to_samples)
        descriptor_ps_lr = descriptor_ps_lr.keys_to_properties(keys_to_move_to_properties)

        # aggregation
        samples_names = ['center', 'species_center', 'spherical_harmonics_l']
        if self._feature_aggregation == 'sum':
            descriptor_co = sum_over_samples(descriptor_co, samples_names=['center'])
            descriptor_ps_sr = sum_over_samples(descriptor_ps_sr, samples_names=samples_names)
            descriptor_ps_lr = sum_over_samples(descriptor_ps_lr, samples_names=samples_names)
        elif self._feature_aggregation == 'mean':
            descriptor_co = mean_over_samples(descriptor_co, samples_names=['center'])
            descriptor_ps_sr = mean_over_samples(descriptor_ps_sr, samples_names=samples_names)
            descriptor_ps_lr = mean_over_samples(descriptor_ps_lr, samples_names=samples_names)

        # joining
        X_sr = join([descriptor_co, descriptor_ps_sr], axis="properties")
        X_lr = join([descriptor_co, descriptor_ps_sr, descriptor_ps_lr], axis="properties")

        return X_lr
