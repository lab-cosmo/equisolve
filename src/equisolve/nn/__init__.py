from .. import HAS_TORCH


if HAS_TORCH:
    from .module_tensor import Linear, ModuleTensorMap  # noqa: F401

    __all__ = ["Linear", "ModuleTensorMap"]
else:
    __all__ = []
