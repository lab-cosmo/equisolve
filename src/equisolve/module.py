from abc import abstractmethod, ABCMeta

from typing import TypeVar

from equistore import TensorMap

# Workaround for typing Self with inheritance for python <3.11
# see https://peps.python.org/pep-0673/
TModule = TypeVar("TModule", bound="Module")

try:
    import torch
    Module = torch.nn.Module
except:
    class Module(metaclass=ABCMeta):
        @abstractmethod
        def forward(self, *args, **kwargs):
            return

        def __call__(self, *args, **kwargs):
            self.forward(*args, **kwargs)


TEstimatorModule = TypeVar("TEstimatorModule", bound="EstimatorModule")

class EstimatorModule(Module, metaclass=ABCMeta):
    def forward(self, X: TensorMap):
        return self.predict(X)

    @abstractmethod
    def fit(self, X: TensorMap, y: TensorMap) -> TEstimatorModule:
        return

    @abstractmethod
    def predict(self, X: TensorMap) -> TensorMap:
        return

    @abstractmethod
    def score(self, X: TensorMap, y: TensorMap) -> TensorMap:
        return

    def fit_score(self, X: TensorMap, y: TensorMap = None) -> TensorMap:
        self.fit(X, y)
        return self.score(X, y)


TTransformerModule = TypeVar("TTransformerModule", bound="TransformerModule")


class TransformerModule(Module, metaclass=ABCMeta):
    def forward(self, X: TensorMap):
        return self.transform(X)

    @abstractmethod
    def fit(self, X: TensorMap, y: TensorMap = None) -> TTransformerModule:
        return

    @abstractmethod
    def transform(self, X: TensorMap) -> TensorMap:
        return

    def fit_transform(self, X: TensorMap, y: TensorMap = None) -> TensorMap:
        self.fit(X, y)
        return self.transform(X)
