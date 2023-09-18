from typing import List, Tuple, Union

import metatensor
from metatensor import TensorMap


try:
    import torch

    HAS_TORCH = True
    TorchModule = torch.nn.Module
except ImportError:
    HAS_TORCH = False
    import abc

    # TODO move to more module.py
    class Module(metaclass=abc.ABCMeta):
        @abc.abstractmethod
        def forward(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        @abc.abstractmethod
        def export_torch(self):
            pass


class AggregateKernel(Module):
    """
    A kernel that aggregates values in a kernel over :param aggregate_names: using
    a aggregaten function given by :param aggregate_type:

    :param aggregate_names:

    :param aggregate_type:
    """

    def __init__(
        self,
        aggregate_names: Union[str, List[str]] = "aggregate",
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
    ):
        valid_aggregate_types = ["sum", "mean"]
        if aggregate_type not in valid_aggregate_types:
            raise ValueError(
                f"Given aggregate_type {aggregate_type!r} but only "
                f"{aggregate_type!r} are supported."
            )
        if structurewise_aggregate:
            raise NotImplementedError(
                "structurewise aggregation has not been implemented."
            )

        self._aggregate_names = aggregate_names
        self._aggregate_type = aggregate_type
        self._structurewise_aggregate = structurewise_aggregate

    def aggregate_features(self, tensor: TensorMap) -> TensorMap:
        if self._aggregate_type == "sum":
            return metatensor.sum_over_samples(
                tensor, samples_names=self._aggregate_names
            )
        elif self._aggregate_type == "mean":
            return metatensor.mean_over_samples(
                tensor, samples_names=self._aggregate_names
            )
        else:
            raise NotImplementedError(
                f"aggregate_type {self._aggregate_type!r} has not been implemented."
            )

    def aggregate_kernel(
        self, kernel: TensorMap, are_pseudo_points: Tuple[bool, bool] = (False, False)
    ) -> TensorMap:
        if self._aggregate_type == "sum":
            if not are_pseudo_points[0]:
                kernel = metatensor.sum_over_samples(kernel, self._aggregate_names)
            if not are_pseudo_points[1]:
                # TODO {sum,mean}_over_properties does not exist
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
                kernel = metatensor.sum_over_properties(kernel, self._aggregate_names)
            return kernel
        elif self._aggregate_type == "mean":
            if not are_pseudo_points[0]:
                kernel = metatensor.mean_over_samples(kernel, self._aggregate_names)
            if not are_pseudo_points[1]:
                # TODO {sum,mean}_over_properties does not exist
                raise NotImplementedError(
                    "properties dimenson cannot be aggregated for the moment"
                )
                kernel = metatensor.mean_over_properties(kernel, self._aggregate_names)
            return kernel
        else:
            raise NotImplementedError(
                f"aggregate_type {self._aggregate_type!r} has not been implemented."
            )

    def forward(
        self,
        tensor1: TensorMap,
        tensor2: TensorMap,
        are_pseudo_points: Tuple[bool, bool] = (False, False),
    ) -> TensorMap:
        return self.aggregate_kernel(
            self.compute_kernel(tensor1, tensor2), are_pseudo_points
        )

    def compute_kernel(self, tensor1: TensorMap, tensor2: TensorMap) -> TensorMap:
        raise NotImplementedError("compute_kernel needs to be implemented.")


class AggregateLinear(AggregateKernel):
    def __init__(
        self,
        aggregate_names: Union[str, List[str]] = "aggregate",
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
    ):
        super().__init__(aggregate_names, aggregate_type, structurewise_aggregate)

    def forward(
        self,
        tensor1: TensorMap,
        tensor2: TensorMap,
        are_pseudo_points: Tuple[bool, bool] = (False, False),
    ) -> TensorMap:
        # we overwrite default behavior because for linear kernels we can do it more
        # memory efficient
        if not are_pseudo_points[0]:
            tensor1 = self.aggregate_features(tensor1)
        if not are_pseudo_points[1]:
            tensor2 = self.aggregate_features(tensor2)
        return self.compute_kernel(tensor1, tensor2)

    def compute_kernel(self, tensor1: TensorMap, tensor2: TensorMap) -> TensorMap:
        return metatensor.dot(tensor1, tensor2)

    def export_torch(self):
        raise NotImplementedError("export_torch has not been implemented")
        # idea is to do something in the lines of
        # return euqisolve.torch.kernels.AggregateLinear(
        #        self._aggregate_names,
        #        self._aggregate_type)


class AggregatePolynomial(AggregateKernel):
    def __init__(
        self,
        aggregate_names: Union[str, List[str]] = "aggregate",
        aggregate_type: str = "sum",
        structurewise_aggregate: bool = False,
        degree: int = 2,
    ):
        super().__init__(aggregate_names, aggregate_type, structurewise_aggregate)
        self._degree = 2

    def compute_kernel(self, tensor1: TensorMap, tensor2: TensorMap):
        return metatensor.pow(metatensor.dot(tensor1, tensor2), self._degree)

    def export_torch(self):
        raise NotImplementedError("export_torch has not been implemented")
        # idea is to do something in the lines of
        # return euqisolve.torch.kernels.AggregatePolynomial(
        #        self._aggregate_names,
        #        self._aggregate_type,
        #        self._degree)
