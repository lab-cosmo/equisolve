import torch
from equisolve.numpy.models import Ridge
import numpy as np
from equisolve.numpy.utils import matrix_to_block, tensor_to_tensormap as tensor_to_numpy_tensormap, tensor_to_torch_tensormap

def equisolve_solver_from_numpy_arrays(
    X_arr, y_arr, alpha_arr, sw_arr=None, solver="auto"
):
    X, y, alpha, sw = to_equistore(X_arr, y_arr, alpha_arr, sw_arr)
    clf = Ridge(parameter_keys="values")
    clf.fit(X=X, y=y, alpha=alpha, sample_weight=sw, solver=solver)
    return clf

def to_equistore(X_arr=None, y_arr=None, alpha_arr=None, sw_arr=None, tensor_to_tensormap=tensor_to_numpy_tensormap):
    """Convert Ridge parameters into equistore Tensormap's with one block."""

    returns = ()
    if X_arr is not None:
        assert len(X_arr.shape) == 2
        X = tensor_to_tensormap(X_arr[None, :])
        returns += (X,)
    if y_arr is not None:
        assert len(y_arr.shape) == 1
        y = tensor_to_tensormap(y_arr.reshape(1, -1, 1))
        returns += (y,)
    if alpha_arr is not None:
        assert len(alpha_arr.shape) == 1
        alpha = tensor_to_tensormap(alpha_arr.reshape(1, 1, -1))
        returns += (alpha,)
    if sw_arr is not None:
        assert len(sw_arr.shape) == 1
        sw = tensor_to_tensormap(sw_arr.reshape(1, -1, 1))
        returns += (sw,)

    if len(returns) == 0:
        return None
    if len(returns) == 1:
        return returns[0]
    else:
        return returns

num_properties = 119
num_targets = 87
solver = "auto"
X = np.random.normal(-0.5, 1, size=(num_targets, num_properties))
w_exact = np.random.normal(-0.5, 3, size=(num_properties,))
y = X @ w_exact
sample_w = np.ones((num_targets,))
property_w = np.zeros((num_properties,))

# Use solver to compute weights from X and y
ridge_class = equisolve_solver_from_numpy_arrays(
    X, y, property_w, sample_w, solver
)
X_tm = tensor_to_torch_tensormap(torch.tensor(X[None, :]))
#y = ridge_class.predict(X_tm)

torch.jit.script(ridge_class)
