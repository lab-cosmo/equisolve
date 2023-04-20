# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import List, Tuple, Dict, TypeVar

from equistore import TensorMap

from ...utils.metrics import rmse

import numpy as np

# Workaround for typing Self with inheritance for python <3.11
# see https://peps.python.org/pep-0673/
TEquiScript = TypeVar("TEquiScript", bound="EquiScript")


class EquiScriptBase(metaclass=ABCMeta):
    """
    An EquiScript is a merge of a representation calculator and a ML model.

    EquiScript supports scikit-learn like transformers and estimators that
    have a fit and predict function and can handle equistore objects.
    For some applications (e.g. long-range interaction, multi-spectra) we want to apply transformers
    seperately on different representations (e.g. because they are on different scales)
    and join them together for the estimation.

    Workflow
    ```python
        hypers = ...
        script = EquiScriptBase(hypers, estimator=Ridge(parameter_keys=["values", "positions"])
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

    def __init__(
        self,
        hypers,
        *,
        feature_aggregation="mean",
        transformer_X=None,
        transformer_y=None, # TODO to a basisclass that ensures that inverse_transform and so an are implemented
        estimator=None
    ):
        self.hypers = hypers
        self.feature_aggregation = feature_aggregation
        self.transformer_X = transformer_X
        self.transformer_y = transformer_y
        self.estimator = estimator

    # FOR ALEX TO REMEMBER the reasoning for private and public member variables 
    # COMMENT it is easier for some classes (MultiSpectra) to assume that the init paramaters are never changed
    #         so we do not allow public setting, this is because unlike scikit estimators we have a compute and a fit function
    #         that both rely on the same member variables.
    #         For example in MultiSpectra: spectra values may not change between compute and fit
    # COMMENT I don't know if just keeping the private ones (e.g. _hypers) is the better solution,
    #         the idea is that if during compute or fit the default values for None objects changes depending on the given arguments,
    #         then one needs to know that the input argument was None, otherwise you overwrite them with one fit or compute call
    # COMMENT I changed code such that _join and fit are completetly independent, so we can actually keep it this way

    def fit(self, X: Dict[str, TensorMap], y: TensorMap, **kwargs) -> TEquiScript:
        """TODO"""
        # X : (X0, X1, ..., XN)
        self._set_and_check_fit_parameters()
        if "transformer_X" not in kwargs.keys():
            kwargs["transformer_X"] = {}
        if "transformer_y" not in kwargs.keys():
            kwargs["transformer_y"] = {}
        if "estimator" not in kwargs.keys():
            kwargs["estimator"] = {}

        X = self._move(X)
        X = self._aggregate(X)

        # TODO move to check and set?
        if self.transformer_X is None:
            self._transformer_X = None
        else:
            if isinstance(self.transformer_X, list):
                if len(self.transformer_X) != len(X):
                    raise ValueError(f"List of transformers and list of descriptors have different length {len(self.transformer_X)} != {len(X)}")
                self.transformer_X = deepcopy(self.transformer_X)
                for i, (descriptor_key, Xi) in enumerate(X.items()):
                    self.transformer_X.append(
                        self.transformer_X[i].fit(Xi, **kwargs["transformer_X"])
                    )
            else:
                self._transformer_X = []
                for descriptor_key, Xi in X.items():
                    self._transformer_X.append(
                        deepcopy(self.transformer_X).fit(Xi, **kwargs["transformer_X"])
                    )

        if self.transformer_y is None:
            self._transformer_y = None
        else:
            self._transformer_y = deepcopy(self.transformer_y).fit(
                y, **kwargs["transformer_y"]
            )

        if self.transformer_X is not None:
            X = {descriptor_key:
                self._transformer_X[i].transform(Xi)
                    for i, (descriptor_key, Xi) in enumerate(X.items())
            }

        X = self._join(X)

        if self.transformer_y is not None:
            y = self._transformer_y.transform(y)


        # TODO move to check and set?
        if self.estimator is None:
            self.estimator = None
        else:
            self.estimator = deepcopy(self.estimator).fit(X, y, **kwargs["estimator"])

    def forward(self, X: Dict[str, TensorMap]) -> TensorMap:
        """TODO"""
        # TODO check if is fitted

        X = self._move(X)
        X = self._aggregate(X)

        if self._transformer_X is not None:
            X = {descriptor_key:
                self._transformer_X[i].transform(Xi)
                    for i, (descriptor_key, Xi) in enumerate(X.items())
            }
        # COMMENT we cannot do this because the join funciton also aggregates at tho moment
        #         maybe we can split this better
        #if len(X) > 1:
        #    # COMMENT I removed the error because it should be thrown in the _join function right?

        X = self._join(X)

        if self.estimator is None:
            return X
        else:
            y_pred = self.estimator.predict(X)
            if self._transformer_y is not None:
                y_pred = self._transformer_y.inverse_transform(y_pred)
            return y_pred

    def score(self, X: Dict[str, TensorMap], y) -> List[float]:
        """TODO"""
        # TODO(low-prio) add support for more error functions
        if self.estimator is None:
            raise ValueError("Cannot use score function without setting an estimator.")

        y_pred =  self.forward(X)

        return np.mean([rmse(y, y_pred, parameter_key) for parameter_key in self.estimator.parameter_keys])


    def _set_and_check_fit_parameters(self) -> None:
        # TODO check if parameter_keys are consistent over the transformers and estimators
        #      and if not throw warning

        if self.feature_aggregation not in ["sum", "mean"]:
            raise ValueError(
                "Only 'sum' and 'mean' are supported for feature_aggregation"
            )

        # TODO would rename to _fit_*
        # we save all member variables, to have the member variables used in the last fit
        self._feature_aggregation = self.feature_aggregation

    @abstractmethod
    def _set_and_check_compute_parameters(self) -> None:
        # COMMENT I dont like naming yet, I mean the parameters relevant for the compute paramaters but not the actual kwargs
        """TODO"""
        raise NotImplemented("Only implemented in child classes")

    @abstractmethod
    def compute(self, **kwargs) -> Tuple[TensorMap, ...]:
        """TODO"""
        self._set_and_check_compute_parameters()
        raise NotImplemented("Only implemented in child classes")

    @abstractmethod
    def _move(self, X: Dict[str, TensorMap]) -> Dict[str, TensorMap]:
        """TODO"""
        raise NotImplemented("Only implemented in child classes")

    @abstractmethod
    def _aggregate(self, X: Dict[str, TensorMap]) -> Dict[str, TensorMap]:
        """TODO"""
        raise NotImplemented("Only implemented in child classes")

    @abstractmethod
    def _join(self, X: Dict[str, TensorMap]) -> TensorMap:
        """TODO"""
        raise NotImplemented("Only implemented in child classes")


