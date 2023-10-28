"""
Computing a SoR Kernel Model
============================

.. start-body

In this tutorial we calculate a kernel model using subset of regressor (SoR)
Kernel model.
"""

import ase.io
import metatensor
from rascaline import SoapPowerSpectrum

from equisolve.numpy.models import SorKernelRidge
from equisolve.numpy.sample_selection import FPS
from equisolve.utils import ase_to_tensormap


frames = ase.io.read("dataset.xyz", ":20")
y = ase_to_tensormap(frames, energy="energy")
n_to_select = 100
degree = 3

HYPER_PARAMETERS = {
    "cutoff": 5.0,
    "max_radial": 6,
    "max_angular": 4,
    "atomic_gaussian_width": 0.3,
    "center_atom_weight": 1.0,
    "radial_basis": {
        "Gto": {},
    },
    "cutoff_function": {
        "ShiftedCosine": {"width": 0.5},
    },
}

calculator = SoapPowerSpectrum(**HYPER_PARAMETERS)

descriptor = calculator.compute(frames, gradients=[])

descriptor = descriptor.keys_to_samples("species_center")
descriptor = descriptor.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])

pseudo_points = FPS(n_to_select=n_to_select).fit_transform(descriptor)

clf = SorKernelRidge()
clf.fit(
    descriptor,
    pseudo_points,
    y,
    kernel_type="polynomial",
    kernel_kwargs={"degree": 3, "aggregate_names": ["center", "species_center"]},
)
y_pred = clf.predict(descriptor)

print(
    "MAE:",
    metatensor.mean_over_samples(
        metatensor.abs(metatensor.subtract(y_pred, y)), "structure"
    )[0].values[0, 0],
)
