# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the BSD 3-Clause "New" or "Revised" License
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

from typing import List, Optional, Union
from equistore import TensorMap

from ...utils.metrics import rmse

class TransformedTargetRegressor:
    r"""Regressor that transforms the target values with a transformer before fitting
        and inverse transforms the predicted target before outputting the score.
        This is useful for doing a regression on a numerical stable scale while still outputting the scores in the original scale.

        :param regressor: The regressor used for doing the prediction.

        :param transformer: The transformer applied on the target y values before regression.
    """

    def __init__(self, regressor=None, transformer=None):
        self.regressor = regressor
        self.transformer = transformer

        if set(regressor.parameter_keys) != set(transformer.parameter_keys):
            raise ValueError("parameter_keys between regressor and transformer do not match")

        if regressor is not None:
            if hasattr(regressor, "fit") and not(callable(regressor.fit)):
                raise ValueError("regressor does not have fit function")
            # TODO more checks for functionalities 

        # COMMENT I am not sure if we should check the regressor and transformer this way,
        #         a base class with defined functionalities seems more clean


        # COMMENT scikit-learn offers also
        #self.func
        #self.inverse_func
        #self.check_inverse
        # but we do not need them at the moment, make low-prio issue to not forget?

    def fit(self, X: TensorMap, y: TensorMap) -> None:
        # COMMENT default args StandardScaler + Ridge? But what parameter keys?
        #if self.regressor is None:
        #    self.regressor_ = Ridge()
        #if self.transformer is None:
        #    self.transformer_ = StandardScaler()

        self.transformer_ = deepcopy(self.transformer).fit(y)
        self.regressor_ = deepcopy(self.regressor).fit(X, self.transformer_.transform(y))
        return self

    def predict(self, X: TensorMap) -> TensorMap:
        return self.transformer_.inverse_transform(self.regressor_.predict(X))

    def score(self, X: TensorMap, y: TensorMap, parameter_key: str) -> float:
        y_pred = self.transformer_.inverse_transform(self.regressor_.predict(X))
        return rmse(y, y_pred, parameter_key)


class Pipeline:
    r"""sklearn style Pipeline

        :param steps: list of (name, transformer/estimator) tuples that are applied in the fit and prediction steps consecutiely in the given order
                      last tuple must contain estimator, all other tuples must contain transformers
                      #COMMENT sklearn Pipeline supports more, but I think this limitation is okay for now
    """
    def __init__(self, steps=None):
        self.steps = steps

    def fit(self, X : TensorMap, y : TensorMap) -> None:
        # TODO check if they have fit function, for this we require base clases for estimator and transformers
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            fitted_transformer = deepcopy(transformer).fit(X, y)
            X = fitted_transformer.transform(X)
            self.steps[step_idx] = (name, fitted_transformer)

        final_estimator = self.steps[-1][1]
        final_estimator.fit(X,y)
        return self

    def predict(self, X: TensorMap) -> TensorMap:
        # TODO check they are fitted, for this we require base clases for estimator and transformers
        for _, transformer in self.steps[:-1]:
            X = transformer.transform(X)
        return self.steps[-1][1].predict(X)

    def score(self, X: TensorMap, y: TensorMap, parameter_key: Optional[str]) -> float:
        # COMMENT we use scorer from last estimator, maybe not what we want
        for _, transformer in self.steps[:-1]:
            X = transformer.transform(X)
        final_estimator = self.steps[-1][1]
        # COMMENT: what do you think about using default parameter_keys?
        if parameter_key is None:
            scores = np.zeros(len(final_estimator.parameter_keys))
            for i, parameter_key in enumerate(final_estimator.parameter_keys):
                scores[i] = final_estimator.score(X, y, parameter_key)[0]
            return np.mean(scores)
        else:
            return final_estimator.score(X, y, parameter_key)
