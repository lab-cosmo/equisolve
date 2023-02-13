# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import List, Tuple, TypeVar

from equistore import TensorMap


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
        script = EquiScriptBase(hypers, parameter_keys=["values", "positions"])
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
        transformer_y=None,
        estimator=None,
    ):
        self.hypers = hypers

        self.feature_aggregation = feature_aggregation

        if estimator is None:
            raise NotImplemented(
                "Ridge still needs a good default alpha value, before we can support default parameter"
            )
            # empty_tm = TensorMap(keys=Labels.single(), blocks=[TensorBlock(
            #    values=np.empty((0,1)),
            #    samples=Labels(names=["samples"], values=np.empty((0,1), dtype=np.int32)),
            #    components=[],
            #    properties=Labels.single(),
            # )])
            # self.estimator = Ridge(parameter_keys=parameter_keys, alpha=empty_tm)

    def fit(self, X: Tuple[TensorMap, ...], y: TensorMap, **kwargs) -> TEquiScript:
        """TODO"""
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
                self._transformers_X.append(
                    deepcopy(self.transformer_X).fit(X[i], **kwargs["transformer_X"])
                )

        if self.transformer_y is None:
            self._transformer_y = None
        else:
            self._transformer_y = deepcopy(self.transformer_y).fit(
                y, **kwargs["transformer_y"]
            )

        X = tuple(
            self._transformers_X[i].transform(X[i])
            for i in range(len(self._transformers_X))
        )
        y = self._transformer_y.transform(y)

        X = self.transform_join(X)

        self._estimator = deepcopy(self.estimator).fit(X, y, **kwargs["estimator"])

    def forward(self, X: Tuple[TensorMap, ...]) -> TensorMap:
        """TODO"""
        # TODO check if is fitted

        if self._transformers_X is not None:
            X = tuple(
                self._transformers_X[i].transform(X[i])
                for i in range(len(self._transformers_X))
            )
        if self._transformer_y is not None:
            y = self._transformer_y.transform(y)

        if len(X) > 1:
            try:
                X = self._join(X)
            except NotImplemented as e:
                raise NotImplemented(
                    "More than one X. But join function is not implemented!"
                ) from e
        else:
            X = X[0]

        y_pred = self._estimator.predict(X, y)
        if self._transformer_y is not None:
            y_pred = self._transformer_y.inverse_transform(y_pred)
        return y_pred

    def score(self, X: Tuple[TensorMap, ...], y) -> List[float]:
        """TODO"""
        # TODO(low-prio) add support for more error functions
        y_pred = self.predict(X, y)
        return self.estimator.rmse(y, y_pred, self._parameter_keys)

    def _set_and_check_fitting_parameters(self) -> None:
        # TODO check if parameter_keys are consistent over the transformers and estimators
        #      and if not throw warning

        if self.feature_aggregation not in ["sum", "meam"]:
            raise ValueError(
                "Only 'sum' and 'mean' are supported for feature_aggregation"
            )

        # TODO would rename to _fit_*
        # we save all member variables, to have the member variables used in the last fit
        self._hypers = self.hypers
        self._feature_aggregation = self._feature_aggregation
        self._parameter_keys = self.parameter_keys

    @abstractmethod
    def _join(self, X: Tuple[TensorMap, ...]) -> TensorMap:
        """TODO"""
        raise NotImplemented("Only implemented in child classes")

    @abstractmethod
    def compute(self, **kwargs) -> Tuple[TensorMap, ...]:
        """TODO"""
        raise NotImplemented("Only implemented in child classes")
