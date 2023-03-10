from typing import Union
from equistore import TensorMap

import equistore


def compute_sparse_kernel(
        tensor: TensorMap,
        pseudo_points: TensorMap,
        degree: int) -> TensorMap:
    equistore.pow(equistore.dot(tensor, pseudo_points), degree)


class SparseKernelRidge:
    def __init__(self):
        pass

    def fit(self,
            k_mm: TensorMap,
            k_nm: TensorMap,
            k_y: TensorMap,
            alpha: Union[float, TensorMap] = 1.0,
            jitter: float=1e-13,
            solver="auto",
            cond: float = None):
        pass

    # check that alpha has the the same samples as k_nm and only one property

    def predict(self):
        pass
