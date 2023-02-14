import ase.io
import numpy as np

class GenericMDCalculator:

    """Generic MD driver for a equi script

    Initialize with equi script JSON and a structure template, and calculate
    energies and forces based on position/cell updates _assuming the
    order and identity of atoms does not change_.
    """
    def __init__(
        self, equi_script_filename, is_periodic, structure_template=None, atomic_numbers=None
    ):
        """Initialize a equi script and structure template

        Parameters
        ----------
        equi_script_filename Filename for the equi script pickle object
        is_periodic Specify whether the simulation is periodic or not
                    This helps avoid confusion if a geometry's "periodic"
                    flags have been set improperly, which can happen e.g.
                    if ASE cannot read the cell information in a file.  If
                    using a structure template and this is set to True,
                    will raise an error unless at least one of the PBC
                    flags in the structure template is on.  If set to
                    False, will raise an error if all PBC flags are not
                    off.  Set to None to skip PBC checking.  If not using a
                    structure template, this setting will determine the PBC
                    flags of the created atomic structure.
        structure_template
                    Filename for an ASE-compatible Atoms object, used
                    only to initialize atom types and numbers
        atomic_numbers
                    List of atom types (atomic numbers) to initialize
                    the atomic structure in case no structure template
                    is given
        """
        self.model_filename = equi_script_filename
        with open(equi_script_filename, "rb") as file:
            self.equi_script = pickle.load(file)
        # Structure initialization
        self.is_periodic = is_periodic
        if structure_template is not None:
            self.template_filename = structure_template
            self.atoms = ase.io.read(structure_template, 0)
            if (is_periodic is not None) and (
                is_periodic != np.any(self.atoms.get_pbc())
            ):
                raise ValueError(
                    "Structure template PBC flags: "
                    + str(self.atoms.get_pbc())
                    + " incompatible with 'is_periodic' setting"
                )
        elif atomic_numbers is not None:
            self.atoms = ase.Atoms(numbers=atomic_numbers, pbc=is_periodic)
        else:
            raise ValueError(
                "Must specify one of 'structure_template' or 'atomic_numbers'"
            )

    def calculate(self, positions, cell_matrix):
        """Calculate energies and forces from position/cell update

        positions   Atomic positions (Nx3 matrix)
        cell_matrix Unit cell (in ASE format, cell vectors as rows)
                    (set to zero for non-periodic simulations)

        The units of positions and cell are determined by the model JSON
        file; for now, only Å is supported.  Energies, forces, and
        stresses are returned in the same units (eV and Å supported).

        Returns a tuple of energy, forces, and stress - forces are
        returned as an Nx3 array and stresses are returned as a 3x3 array

        Stress convention: The stresses have units eV/Å^3
        (volume-normalized) and are defined as the gradients of the
        energy with respect to the cell parameters.
        """
        # Quick consistency checks
        if positions.shape != (len(self.atoms), 3):
            raise ValueError(
                "Improper shape of positions (is the number of atoms consistent?)"
            )
        if cell_matrix.shape != (3, 3):
            raise ValueError("Improper shape of cell info (expected 3x3 matrix)")

        # Update ASE Atoms object (we only use ASE to handle any
        # re-wrapping of the atoms that needs to take place)
        self.atoms.set_cell(cell_matrix)
        self.atoms.set_positions(positions)

        Xi = script.compute(systems=structure, gradients=["positions"])
        y_pred = script.forward(Xi) # implicitely done in score function
        energy = y_pred.block().values[0][0]
        forces = np.array(y_pred.block().gradient("positions").data.reshape(-1, 3))

        # Not implemented
        stress_matrix = np.zeros((3, 3))
        return energy, forces, stress_matrix
