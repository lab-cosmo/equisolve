import equisolve

import ase

from typing import Union

import torch # TODO has to be made optional
import numpy as np

from ..utils import ase_to_tensormap, properties_to_tensormap
from ..module import Module

from equistore import TensorBlock, TensorMap

# PR COMMENT: Temporary class should be replaced by PairPotential
class EquistorePairPotential(Module):
    """
    For models that take as input a TensorMap

    """
    def __init__(self,
                 model : Module,
                 md_style: str, # ["ase-frames", "openkim", "auto"]
                 # PR COMMENT: Why not always use auto? 
                 #             Maybe there are cases where we cannot deduce it automatically.
                 # PR COMMENT we need a units library that is compatible with TorchScript
                 energy_units=None,
                 forces_units=None,
                 stress_units=None):
        supported_md_styles = ["ase-frames", "openkim", "auto"]
        if md_style not in supported_md_styles:
            raise ValueError(f"MD style {md_style} is not supported. We support {supported_md_styles}.")
        self._md_style = md_style
        self._model = model
        self._prediction = None

    def forward(self, *args, **kwargs):
        if self._md_style == "auto":
            # PR COMMENT: for now i focus on the specific branches
            raise NotImplemented("auto is not implemented yet")
            #if len(args) == 1 and isinstance(args[0], "ase.Atoms")
            #    # ase-frames
            #    X = ase_to_tensormap(args[0])
            #    self._prediction = self._model.forward(X)
            #elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
            #    # i-pi
            #    ase.Atoms(positions=args[1], cell=args[0])
            #    self._prediction = self._model.forward(args[0])
            #else:
            #    raise ValueError(f"args {args} could not be automatically detected. Might be not supported")
        elif structure_input_style == "ase-frames":
            X = ase_to_tensormap(args[0])
            self._prediction = self._model.forward(X)
        else:
            NotImplemented(f"MD style {self._md_style} not implemented")

    @property
    def energy(self):
        if self._prediction is None:
            # maybe raise error?
            return None
        energies = self._prediction.block().values
        # do some additional stuff to have coherent units
        return energies

    @property
    def forces(self):
        if self._prediction is None:
            # maybe raise error?
            return None
        forces = self._prediction.block().gradient("positions")
        # do some additional stuff to have coherent units
        return forces

    @property
    def virial_stress(self):
        if self._prediction is None:
            # maybe raise error?
            return None
        virial_stress = self._prediction.block().gradient("cell")
        # do some additional stuff to have coherent units
        return virial_stress


# PR COMMENT: 
#   This class not working at all, it is easier to first get
#   my head around the EuqistorePotential class
#   to get then back to this class
class PairPotential(Module):
    """
    Should cover everything that is based on a pair neighbourlist with a cutoff

    Basic use of this class
    MD code's neighbourlist for pair potential ---PairPotential--> ML models inputs neighbourlist
    ML models output properties ---PairPotential--> MD code's properties format

    with raw I mean a list of the arrays [positions, atom_types, ...]
    better name needs to be found.

    One stil needs interfaces for each MD code, but they can be very thin wrappers.

    """
    def __init__(self,
                 model,
                 cutoff,

                 # PR COMMENT: might be useful to differ between input and output
                 md_style: str, # ["i-pi", "ase-calculator", "open_kim"]

                 model_style: str, # ["raw", "equistore", "ase-frames"]

                 energy_units=None, # what units library to use? i feel like every library has its own shitty one
                 forces_units=None,
                 stress_units=None):
        self.model = model
        # is needed to do a parameter check with the MD code
        self.cutoff = cutoff


    def compute(self, *args, **kwargs):
        if structure_input_style == "ase":
            self._compute_properties_ase(*args, **kwargs)
        elif structure_input_style == "raw":
            self._compute_properties_raw(*args, **kwargs)

    # this is just for python driver compatible MD codes
    def _compute_properties_ase(self, frame: ase.Atoms, atomic_properties=False):
        self.output = model.forward(frame)

    # PR COMMENT:
    #   Should be supported by most NN like SchNet, Allegro 
    #   Probably needs separation for different input types
    def compute_properties_raw(self,
                              positions: Union[TensorBlock, torch.Tensor, np.ndarray],
                              atom_types: Union[TensorBlock, torch.Tensor, np.ndarray],
                              cell: Union[TensorBlock, torch.Tensor, np.ndarray],
                              atomic_properties=False):
        pass


# PR COMMENT:
#   This class needs much more thoughts to be put in.
#   So far I just want to state that for computing
#   long range more efficiently we need to make an 
#   interface that works on the kgrid, so the model returns dE/dk_i
#   https://github.com/lammps/lammps/blob/fce1f8e0af22106abece97c8099815e97c8980c6/src/KSPACE/ewald.cpp#L391
#   but for the MD code we need to use the chain rule to obtain dE/dr_j
class KspacePotential(Module):
    """For long range potentials that work on k-space"""
    pass
