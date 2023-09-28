try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from .module_tensor import Linear, ModuleTensorMap  # noqa: F401

    __all__ = ["Linear", "ModuleTensorMap"]
else:
    __all__ = []
