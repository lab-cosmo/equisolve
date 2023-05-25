"""
Computing a Linear Model
========================

.. start-body

In this tutorial we calculate a linear model using Ridge regression.
If you are never worked with equistore objects before please take a look at
the documentation.

For constructing a linear Model we need the atomic descriptor as training data
``X`` as well as the energies and forces as target data ``y``.

We first import all necessary packages.
"""

import ase.io
import equistore
import numpy as np
from rascaline import SoapPowerSpectrum

from equisolve.numpy.models.linear_model import Ridge
from equisolve.utils.convert import ase_to_tensormap


# %%
#
# Dataset
# -------
#
# As data set we use the SHIFTML set. You can obtain the dataset used in this
# example from our :download:`website<../../static/dataset.xyz>`.
# We read the first 20 structures of the data set using
# `ASE <https://wiki.fysik.dtu.dk/ase/>`_.


frames = ase.io.read("dataset.xyz", ":20")

# %%
#
# The data set contains everything we need for the model:
# The atomic positions we use for the descriptor and with this as
# training data. The data set also stores the energies and forces which will be our
# target data we regress against.
#
# Training data
# -------------
#
# We construct the descriptor training data with a SOAP powerspectrum using
# rascaline. We first define the hyper parameters for the calculation


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

# %%
#
# And then run the actual calculation, including gradients with respect to positions.

descriptor = calculator.compute(frames, gradients=["positions"])

# %%
#
# For more details on how the descriptor works see the documentation of
# rascaline.
#
# We now move all keys into properties to access them for our model.

descriptor = descriptor.keys_to_samples("species_center")
descriptor = descriptor.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])

# %%
#
# The descriptor contains a represenantion with respect to each central atoms per
# structure. However, our energies as target data is per structure only.
# Therefore, we sum the properties of each center atom per structure.

X = equistore.sum_over_samples(descriptor, ["center", "species_center"])

# %%
#
# The newly defined :class:`equistore.TensorMap` contains a single block

print(f"X contains {len(X.blocks())} block.")

# %%
#
# As well as 1800 properties and 20 sample.
#
# We acces the data using the :meth:``equistore.TensorMap.block`` method


print(f"X contains {len(X[0].properties)} properties.")
print(f"X contains {len(X[0].samples)} samples.")

# Target data
# -----------
#
# We construct the target data by converting energies and forces into a
# :class:`equisolve.TensorMap`.


y = ase_to_tensormap(frames, energy="energy", forces="forces")

# %%
#
# The target data y contains a single block

print(y[0])

# %%
#
# Construct the model
# -------------------
#
# We first initilize the :class:`equisolve.numpy.models.linear_model.Ridge`
# object. A mandatory parameter are the ``parameter_keys`` determining with
# respect to which parameters the regression or fit is
# performed. Here, we choose a regression wrt. to ``"values"`` (energies) and
# ``"positions"`` (forces).


clf = Ridge(parameter_keys=["values", "positions"])

# %%
#
# Before we fit a model we have to define our regulerizer values.
#
# For this we create a TensorMap containing the desired regulerizer values.
# Here we chose a regulerizer strength of :math:`1 \cdot 10^-5`. Note that
# without standardizing the features and values the regulerizer strength
# depends on the system and has to be taken carefully and usually optimized.

alpha = equistore.ones_like(X)
alpha = equistore.multiply(alpha, 1e-5)

# %%
#
# So far ``alpha`` contains the same number of samples as ``X``. However,
# the regulerizer must only contain a single sample, because all samples will be
# regulerized in the same way in a linear model.
#
# We remove all sample except the 0th one by using the
# :func:`equistore.slice`.

samples = equistore.Labels(
    names=["structure"],
    values=np.array([(0,)]),
)

alpha = equistore.slice(alpha, axis="samples", labels=samples)

print(alpha)

# %%
#
# In our regulerizer we use the same values for all properties. However,
# :class:`equisolve.numpy.models.linear_model.Ridge` can also handle different
# regularization for each property. You can apply a property wise
# regularization by setting ``"values"`` of ``alpha`` with an 1d array of the
# same length as the number of properties in the training data X (here 7200).
#
# Next we create a sample weighting :class:`equistiore.TensorMap` that weights
# energies five times more then the forces.

sw = equistore.ones_like(y)
sw = equistore.multiply(sw, 5.0)

# %%
#
# The function `equisolve.utils.dictionary_to_tensormap` create a
# :class:`equistore.TensorMap` with the same shape as our target data ``y`` but
# with values a defined by ``sw_dict``.

print(sw[0])

# %%
#
# Finally, we can fit the model using the regulerizer and sample weights as
# defined above.

clf.fit(X, y, alpha=alpha, sample_weight=sw)

# %%
#
# We now predict values and calculate the root mean squre error
# of our model using the ``score`` method.

print(f"RMSE energies = {clf.score(X, y, parameter_key='values')[0]:.3f} eV")
print(f"RMSE forces = {clf.score(X, y, parameter_key='positions')[0]:.3f} eV/Å")

# %%
#
# If you only want to predict values you can use the
# :meth:`equisolve.numpy.models.linear_model.Ridge.predict` method.

clf.predict(X)
