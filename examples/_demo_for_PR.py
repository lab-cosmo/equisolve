import numpy as np
import ase
from equistore import TensorBlock, TensorMap, Labels
from equisolve.vanilla.preprocessing import property_to_tensormap, energy_to_tensormap

# artificial data

n_strucs = 2
energies = np.random.rand(n_strucs)
forces = np.random.rand(n_strucs, 5, 3)

frames = []
for i in range(len(energies)):
    frame = ase.Atoms('CH4')
    frame.info['energy'] = energies[i]
    frame.arrays['forces'] = forces[i]
    frames.append( frame )

property_tm = energy_to_tensormap(frames, 'energy', 'forces')

# tensor map out of one block, because we don't have any components
#print("energies shape", property_tm[0].values.shape)
#print("energies\n", property_tm[0].values)

# we have n_samples x 3 x 1, the last one dimension comes from the energy property
#print("forces shape", property_tm[0].gradient("positions").data.shape)
#print("forces\n", property_tm[0].gradient("positions").data)

# using the property_to_tensormap function with forces, does not do the automatic negation of forces
property_tm2 = property_to_tensormap(frames, 'energy', 'forces')
print("energy_to_tensormap(frames, 'energy', 'forces') == -property_to_tensormap(frames, 'energy', 'forces') is", \
        np.allclose(property_tm[0].gradient("positions").data, property_tm2[0].gradient("positions").data))

get_forces = lambda frame : frame.arrays['forces']
property_tm = energy_to_tensormap(frames, 'energy', get_forces)
get_positions_gradient = lambda frame : -frame.arrays['forces']
property_tm2 = property_to_tensormap(frames, 'energy', get_positions_gradient)
print("energy_to_tensormap(frames, 'energy', 'forces') == property_to_tensormap(frames, 'energy', lambda frame : -frame.arrays['forces']) is", \
        np.allclose(property_tm[0].gradient("positions").data, property_tm2[0].gradient("positions").data))
