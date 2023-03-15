from abc import abstractmethod

from typing import TypeVar

from equistore import TensorMap

# Workaround for typing Self with inheritance for python <3.11
# see https://peps.python.org/pep-0673/
TModule = TypeVar("TModule", bound="Module")

try:
    import torch
    Module = torch.nn.Module
except:
    class Module:
        @abstractmethod
        def forward(self, X: TensorMap):
            raise NotImplemented("Abstract method was invoked, needs implementation.")


TEstimatorModule = TypeVar("TEstimatorModule", bound="EstimatorModule")


class EstimatorModule(Module):
    def forward(self, X: TensorMap):
        return self.predict(X)

    @abstractmethod
    def fit(self, X: TensorMap, y: TensorMap) -> TEstimatorModule:
        raise NotImplemented("Abstract method was invoked, needs implementation.")

    @abstractmethod
    def predict(self, X: TensorMap) -> TensorMap:
        raise NotImplemented("Abstract method was invoked, needs implementation.")

    @abstractmethod
    def score(self, X: TensorMap, y: TensorMap) -> TensorMap:
        raise NotImplemented("Abstract method was invoked, needs implementation.")

    def fit_score(self, X: TensorMap, y: TensorMap = None) -> TensorMap:
        self.fit(X, y)
        return self.score(X, y)


TTransformerModule = TypeVar("TTransformerModule", bound="TransformerModule")


class TransformerModule(Module):
    def forward(self, X: TensorMap):
        return self.predict(X)

    @abstractmethod
    def fit(self, X: TensorMap, y: TensorMap = None) -> TTransformerModule:
        raise NotImplemented("Abstract method was invoked, needs implementation.")

    @abstractmethod
    def transform(self, X: TensorMap) -> TensorMap:
        raise NotImplemented("Abstract method was invoked, needs implementation.")

    def fit_transform(self, X: TensorMap, y: TensorMap = None) -> TensorMap:
        self.fit(X, y)
        return self.transform(X)
