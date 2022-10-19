import ase.io
import urllib.request
import numpy as np
import equistore.io
from equistore import Labels, TensorBlock, TensorMap
import rascaline

dataset = "https://raw.githubusercontent.com/Luthaf/rascaline/858d2c2ba286019d1bb263535661412325c50b12/docs/static/datasets.xyz"
structures_fn, _ = urllib.request.urlretrieve(dataset)
frames = ase.io.read(structures_fn, ":10")
calculator = rascaline.SphericalExpansion(
    cutoff=3.5,
    max_radial=6,
    max_angular=6,
    atomic_gaussian_width=0.3,
    radial_basis={"Gto": {}},
    center_atom_weight=1.0,
    cutoff_function={"ShiftedCosine": {"width": 0.5}},
)
descriptor = calculator.compute(frames)
descriptor.keys_to_samples("species_center")
descriptor.keys_to_properties("species_neighbor")
equistore.io.save("spherical-expansion.npz", descriptor)
calculator = rascaline.SoapPowerSpectrum(
    cutoff=3.5,
    max_radial=6,
    max_angular=6,
    atomic_gaussian_width=0.3,
    radial_basis={"Gto": {}},
    center_atom_weight=1.0,
    cutoff_function={"ShiftedCosine": {"width": 0.5}},
)
descriptor = calculator.compute(frames, gradients=["positions"])
descriptor.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])

equistore.io.save("power-spectrum.npz", descriptor)


energy = []
forces = []

for frame in frames:
    energy.append(frame.info["energy"])
    forces.append(frame.arrays["forces"])

block = TensorBlock(
    values=np.vstack(energy),
    samples=Labels(
        names=["structure"],
        values=np.array([[s] for s in range(len(frames))], dtype=np.int32),
    ),
    components=[],
    properties=Labels(names=["energy"], values=np.array([[0]], dtype=np.int32)),
)
block.add_gradient(
    "positions",
    data=-(np.vstack(forces).reshape(-1, 3, 1)),
    samples=Labels(
        names=["sample", "structure", "center"],
        values=np.array(
            [[s, s, c] for s, f in enumerate(frames) for c in range(len(f))],
            dtype=np.int32,
        ),
    ),
    components=[
        Labels(names=["direction"], values=np.array([[0], [1], [2]], dtype=np.int32)),
    ],
)

energies = TensorMap(Labels.single(), [block])
equistore.io.save("energies.npz", energies)
