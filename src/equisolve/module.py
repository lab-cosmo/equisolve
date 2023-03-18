import pickle

from abc import abstractmethod, ABCMeta

from typing import TypeVar, Dict
from collections import OrderedDict

from equistore import TensorMap, TensorBlock

# Workaround for typing Self with inheritance for python <3.11
# see https://peps.python.org/pep-0673/

class NumpyModule(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, *args, **kwargs):
        return

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    #def __setattr__(self, name, value):
    #    # TODO used to keep track of the parameters that
    #           have been set up
    #    # https://stackoverflow.com/a/54994818
    #    return

    def state_dict(self) -> OrderedDict:
        """
        All required parameteres to initialize a fitted module
        """

        # PR COMMENT: not implemented, but should be analogous
        #             to torch.nn.Module by using __setattr_
        return

    # PR comment: Torch does not return a boolean but a
    #             torch.nn.modules.module._IncompatibleKeys
    #             need to think about this, if we keep bool
    #             or make something analogous
    def load_state_dict(self, state_dict: OrderedDict) -> bool:
        """
        Initialize a fitted module
        """
        # PR COMMENT: not implemented, but should be analogous
        #             to torch.nn.Module by using __setattr_
        return

try:
    import torch
    HAS_TORCH = True
    Module = torch.nn.Module
except:
    HAS_TORCH = False
    Module = NumpyModule


# PR COMMENT:
# We want to allow people to switch to a torch model if needed
# (to use the TorchScript for low level integration into MD code).
# For that we need both class definitions (numpy and torch)
# so we can convert a NumpyModule to a torch Module
# the use case I see is that you did not work with torch
# then you have a class based on NumpyModule, but you need
# a class based on TorchModule, so we provide an utility function
# that takes all the parameters (state_dict) of the numpy module,
# converts them to torch tensors, and reinitialize it with 
# the corresponding torch module

# all modules that have a corresponding torch module
# are stored here
NUMPY_TO_TORCH_MODULE = {}

# PR COMMENT I think this cannot be hiden a class decorator like

#              # automatically creates EstimatorModuleNumpy and EstimatorModuleTorch 
#              @equisolve.module
#              class EstimatorModule(Module)
#                  ...

#            because we would need to change the base class
#            and that is super hacky https://stackoverflow.com/a/9541560

# PR COMMENT all modules that support both would need to be build like this
def estimator_module_factory(base_class, name):
    class _EstimatorModule(base_class, metaclass=ABCMeta):
        def __init__(self):
            super().__init__()

        def forward(self, X: TensorMap):
            return self.predict(X)

        @abstractmethod
        def fit(self, X: TensorMap, y: TensorMap) -> base_class:
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

    # PR COMMENT: this is kind of the idea 
    #             this can be solved more Vnicely, currently the class stored in
    #             equisolve.module.estimator_module_factory.<locals>.EstimatorModule
    #             we can replace https://stackoverflow.com/q/681953 
    _EstimatorModule.__name__ = name
    _EstimatorModule.__qualname__ = name
    return _EstimatorModule

EstimatorNumpyModule = estimator_module_factory(NumpyModule, "EstimatorModuleNumpy")
if HAS_TORCH:
    EstimatorTorchModule = estimator_module_factory(torch.nn.Module, "EstimatorModuleTorch")
    NUMPY_TO_TORCH_MODULE["EstimatorNumpyModule"] = EstimatorTorchModule
    # this is just reference to the default module type
    EstimatorModule = EstimatorTorchModule
else:
    # this is just reference to the default module type
    EstimatorModule = EstimatorNumpyModule


# PR COMMENT: for now I did not adapt the transformer module
#             it is equivalent, but not worth the effort if we dont follow approach
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

def save(module: Module, f: str):
    """
    Saves to a pickable object
    """
    # PR COMMMENT: 
    # torch.save is just nice wrapper for pickle as far as I understand.
    # It can also save TorchScripts by dispatching to jit.save,
    # but that is not really intended. So we can just use it here
    if HAS_TORCH:
        torch.save(f)
    else:
        with open(f, 'wb') as file:
            module = pickle.dump(file)


def load(f: str):
    """
    Loads a pickable object
    """
    if HAS_TORCH:
        torch.load(f)
    else:
        with open(f, 'rb') as file:
            module = pickle.load(file)

if HAS_TORCH:
    # PR COMMENT: Maybe we can store also the init args and kwargs
    #             in our models so we can get them from module input
    #             torch does not do it as far as I have seen, so 
    #             I did not implement it so far
    def convert_to_torch_module(module: NumpyModule, *module_init_args, **module_init_kwargs):
        """
        converts a numpy module to a torch module
        """
        module.state_dict()
        if module.__class__.__name__ not in NUMPY_TO_TORCH_MODULE.keys():
            raise NotImplemented(f"Your module {module.__class__.__name__} "
                                  "has not been implemented as torch module.")
        torch_module_class = NUMPY_TO_TORCH_MODULE(module.__class__.__name__)
        torch_module = torch_module_class(*module_init_args, **module_init_kwargs)
        state_dict_numpy = module.state_dict()
        state_dict_torch = {}
        # PR COMMENT: that is the idea, this code is not working, 
        #             also this might a bit more complicated if nested objects exist
        #             need to read more in detail how state_dict works
        #             https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html
        for key, arg in state_dict_numpy.items():
            if isinstance(arg, TensorMap):
                state_dict_torch[key] = argument.to_tensor()
            if isinstance(arg, TensorBlock):
                state_dict_torch[key] = argument.to_tensor()
        torch_module.load_state_dict(state_dict_torch)
        return torch_module

def save_torch_script(module: Module):
    # traces module saves it
    if not(has_torch):
        raise importerror("saving a model as torchscript requires torch import.")
    if issubclass(module, torch.nn.module):
        torch.jit.save(module)
    elif issubclass(module, Module):
        raise ValueError(f"Your module of type {module.__class__} is not a torch module and cannot be saved as a torch script. "
                          "Please convert your module first using convert_to_torch_module.")
    else:
        raise ValueError(f"Your module of type {module.__class__} is not a torch module and cannot be saved as a torch script.")

def load_torch_script(f: str):
    return torch.jit.load(f)
