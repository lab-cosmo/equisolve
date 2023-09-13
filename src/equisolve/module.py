from abc import ABCMeta, abstractmethod
from typing import TypeVar

from metatensor import TensorMap


class NumpyModule(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, *args, **kwargs):
        return

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


T_Estimator = TypeVar("_Estimator", bound="_Estimator")


class _Estimator(metaclass=ABCMeta):
    def forward(self, X: TensorMap):
        return self.predict(X)

    @abstractmethod
    def fit(self, X: TensorMap, y: TensorMap) -> T_Estimator:
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


T_Transformer = TypeVar("_Transformer", bound="_Transformer")


class _Transformer(metaclass=ABCMeta):
    def forward(self, X: TensorMap):
        return self.transform(X)

    @abstractmethod
    def fit(self, X: TensorMap, y: TensorMap = None) -> T_Transformer:
        return

    @abstractmethod
    def transform(self, X: TensorMap) -> TensorMap:
        return

    def fit_transform(self, X: TensorMap, y: TensorMap = None) -> TensorMap:
        self.fit(X, y)
        return self.transform(X)
