import numpy as np
import ase
from equistore import TensorBlock, TensorMap, Labels
from equisolve.vanilla.preprocessing import ase_frames_to_properties_tensormap

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

get_energy = lambda frame : frame.info['energy']
get_forces = lambda frame : frame.arrays['forces']
properties_tm = ase_frames_to_properties_tensormap(frames, get_energy, get_forces)

# tensor map out of one block, because we don't have any components
print("energies shape", properties_tm[0].values.shape)
print("energies\n", properties_tm[0].values)

# we have n_samples x 3 x 1, the last one dimension comes from the energy property
print("forces shape", properties_tm[0].gradient("positions").data.shape)
print("forces\n", properties_tm[0].gradient("positions").data)
