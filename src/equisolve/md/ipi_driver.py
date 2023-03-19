from ..module import load
import ase.io
from ipi.utils.units import unit_to_internal, unit_to_user
import sys

# shoud be in i-pi
class IpiDriver:
    def __init__(self, args_string : str = None):
        """
        args_string : concatenated string passed by the user
        """
        try:
            args_list = args_string.split(",")
        except ValueError:
            sys.exit(f"Something went wrong with equistore potential when splitting the {args_string}")

        if len(args_list) != 2:
            sys.exit(f"Equistore potential requires 2 arguments but {len(args_list)} were given: {args_list}")

        self._potential_filename = args_list[0]

        # reference structure is needed to determine atomic numbers,
        # because i-pi does not pass them, smart i-pi
        self._reference_structure = ase.io.read(args_list[1])

        self._potential = load(self._potential_filename)

    def __call__(self, cell, pos):
        # PR comment: this unit conversion should be probably moved 
        #             to the PairPotential class
        pos_ = unit_to_user("length", "angstrom", pos)
        cell_ = unit_to_user("length", "angstrom", cell.T)

        frame = ase.Atoms(positions=pos, cell=cell)
        frame.numbers = self._reference_structure.numbers

        self._potential.compute(frame)

        potential_energies = unit_to_internal("energy", "electronvolt",
                                   self._potential.energies.copy())
        forces = unit_to_internal("force", "ev/ang",
                                  self._potential.forces.copy())

        # virial stress not supported ATM
        extras = ""
        return energies, forces, np.zeros((3, 3)), extras
