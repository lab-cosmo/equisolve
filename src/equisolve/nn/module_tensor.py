from .. import HAS_METATENSOR_TORCH


if HAS_METATENSOR_TORCH:
    from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap
else:
    from metatensor import Labels, LabelsEntry, TensorBlock, TensorMap

from copy import deepcopy
from typing import List, Optional

import torch
from torch.nn import Module, ModuleDict


@torch.jit.interface
class ModuleTensorMapInterface(torch.nn.Module):
    """
    This interface required for TorchScript to index the :py:class:`torch.nn.ModuleDict`
    with non-literals in ModuleTensorMap. Any module that is used with ModuleTensorMap
    must implement this interface to be TorchScript compilable.

    Note that the *typings and argument names must match exactly* so that an interface
    is correctly implemented.

    Reference
    ---------
    https://github.com/pytorch/pytorch/pull/45716
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass


class ModuleTensorMap(Module):
    """
    A wrapper around a :py:class:`torch.nn.ModuleDict` to apply each module to the
    corresponding tensor block in the map using the dict key.

    :param module_map:
        A dictionary of modules with tensor map keys as dict keys
        each module is applied on a block

    :param out_tensor:
        A tensor map that is used to determine the properties labels of the output.
        Because an arbitrary module can change the number of properties, the labels of
        the properties cannot be persevered. By default the output properties are
        relabeled using Labels.range.
    """

    def __init__(self, module_map: ModuleDict, out_tensor: Optional[TensorMap] = None):
        super().__init__()
        self._module_map = module_map
        # copy to prevent undefined behavior due to inplace changes
        if out_tensor is not None:
            out_tensor = out_tensor.copy()
        self._out_tensor = out_tensor

    @classmethod
    def from_module(
        cls,
        in_keys: Labels,
        module: Module,
        many_to_one: bool = True,
        out_tensor: Optional[TensorMap] = None,
    ):
        """
        A wrapper around one :py:class:`torch.nn.Module` applying the same type of
        module on each tensor block.

        :param in_keys:
            The keys that are assumed to be in the input tensor map in the
            :py:meth:`forward` function.
        :param module:
            The module that is applied on each block.
        :param many_to_one:
            Specifies if a separate module for each block is used. If `True` the module
            is deep copied for each key in the :py:attr:`in_keys`.
        :param out_tensor:
            A tensor map that is used to determine the properties labels of the output.
            Because an arbitrary module can change the number of properties, the labels
            of the properties cannot be persevered. By default the output properties are
            relabeled using Labels.range.
        """
        module = deepcopy(module)
        module_map = ModuleDict()
        for key in in_keys:
            module_key = ModuleTensorMap.module_key(key)
            if many_to_one:
                module_map[module_key] = module
            else:
                module_map[module_key] = deepcopy(module)

        return cls(module_map, out_tensor)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Takes a tensor map and applies the modules on each key it.

        :param tensor:
            input tensor map
        """
        out_blocks: List[TensorBlock] = []
        for key, block in tensor.items():
            out_block = self.forward_block(key, block)

            for parameter, gradient in block.gradients():
                if len(gradient.gradients_list()) != 0:
                    raise NotImplementedError(
                        "gradients of gradients are not supported"
                    )
                out_block.add_gradient(
                    parameter=parameter,
                    gradient=self.forward_block(key, gradient),
                )
            out_blocks.append(out_block)

        return TensorMap(tensor.keys, out_blocks)

    def forward_block(self, key: LabelsEntry, block: TensorBlock) -> TensorBlock:
        module_key: str = ModuleTensorMap.module_key(key)
        module: ModuleTensorMapInterface = self._module_map[module_key]
        out_values = module.forward(block.values)
        if self._out_tensor is None:
            properties = Labels.range("_", out_values.shape[-1])
        else:
            properties = self._out_tensor.block(key).properties
        return TensorBlock(
            values=out_values,
            properties=properties,
            components=block.components,
            samples=block.samples,
        )

    @property
    def module_map(self):
        """
        The :py:class:`torch.nn.ModuleDict` that maps hashed module keys to a module
        (see :py:func:`ModuleTensorMap.module_key`)
        """
        # type annotation in function signature had to be removed because of TorchScript
        return self._module_map

    @property
    def out_tensor(self) -> Optional[TensorMap]:
        """
        The tensor map that is used to determine properties labels of the output of
        forward function.
        """
        return self._out_tensor

    @staticmethod
    def module_key(key: LabelsEntry) -> str:
        return str(key)


class Linear(ModuleTensorMap):
    """
    :param in_tensor:
        A tensor map that will be accepted in the :py:meth:`forward` function. It is
        used to determine the keys input shape, device and dtype of the input to create
        linear modules for tensor maps.

    :param out_tensor:
        A tensor map that is used to determine the properties labels and shape of the
        output tensor map.  Because a linear module can change the number of
        properties, the labels of the properties cannot be persevered.

    :param bias:
        See :py:class:`torch.nn.Linear`
    """

    def __init__(
        self,
        in_tensor: TensorMap,
        out_tensor: TensorMap,
        bias: bool = True,
    ):
        module_map = ModuleDict()
        for key, in_block in in_tensor.items():
            module_key = ModuleTensorMap.module_key(key)
            out_block = out_tensor.block(key)
            module = torch.nn.Linear(
                len(in_block.properties),
                len(out_block.properties),
                bias,
                in_block.values.device,
                in_block.values.dtype,
            )
            module_map[module_key] = module

        super().__init__(module_map, out_tensor)

    @classmethod
    def from_module(
        cls,
        in_keys: Labels,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        many_to_one: bool = True,
        out_tensor: Optional[TensorMap] = None,
    ):
        """
        :param in_keys:
            The keys that are assumed to be in the input tensor map in the
            :py:meth:`forward` function.
        :param in_features:
            See :py:class:`torch.nn.Linear`
        :param out_features:
            See :py:class:`torch.nn.Linear`
        :param bias:
            See :py:class:`torch.nn.Linear`
        :param device:
            See :py:class:`torch.nn.Linear`
        :param dtype:
            See :py:class:`torch.nn.Linear`
        :param many_to_one:
            Specifies if a separate module for each block is used. If True the module is
            deepcopied for each key in the :py:attr:`in_keys`.
        :param out_tensor:
            A tensor map that is used to determine the properties labels of the output.
            Because an arbitrary module can change the number of properties, the labels
            of the properties cannot be persevered. By default the output properties are
            relabeled using Labels.range.
        """
        module = torch.nn.Linear(in_features, out_features, bias, device, dtype)
        return ModuleTensorMap.from_module(in_keys, module, many_to_one, out_tensor)

    def forward(self, tensor: TensorMap) -> TensorMap:
        # added to appear in doc, :inherited-members: is not compatible with torch
        return super().forward(tensor)
