import numpy as np

from equistore import TensorMap, Labels
from equistore.block import TensorBlock, Gradient

def ase_frames_to_properties_tensormap(frames, get_energy, get_forces=None, get_viral_stress=None):
    """
    Creates TensorMap from ase frames containing atomic properties

    :param frames:
    :param get_energy: function that returns float
        function that extracts energy from frame

        ```python
            energy = get_energy(frame)
        ```

    :param get_forces: function that returns array of shape (n_atoms, 3)
        function that extracts forces from frame

        ```python
            forces = get_forces(frame)
        ```

    """
    if get_viral_stress is not None:
        raise NotImplemented("Viral stress is not supported at the moment")

    if not isinstance(frames, list):
        frames = [frames]

    #all_species = set()
    #for frame in frames:
    #    all_species.update(frame.numbers)

    keys = Labels(
        names=["idx"],
        values=np.array([[0]], dtype=np.int32)
    )

    energies = np.array([get_energy(frame) for frame in frames])
    # -1 dimension because of property dimension
    forces = np.concatenate([get_forces(frame) for frame in frames], axis=0).reshape(-1, 3, 1)

    samples_energies = Labels(
        names=["structure"],
        values=np.array([
            [s]
            for s in range(len(frames))
        ], dtype=np.int32)
    )

    # gradients needs sample dimension
    offsets = np.array([0] + [len(frame) for frame in frames])
    samples_gradients = Labels(
        names=["sample", "structure", "atom"],
        values=np.array([
            [offsets[s]+a, s, a] 
            for s in range(len(frames))
            for a in range(len(frames[s]))
        ], dtype=np.int32)
    )

    blocks = [TensorBlock(
        values=samples_energies.reshape(-1,1),
        samples=samples_energies,
        components=[],
        properties=Labels(['energy'], np.array([(0,)], dtype=np.int32))
    )]

    gradients_labels = Labels(
        ['direction'],
        np.array([(0,), (1,), (2,)], dtype=np.int32)
    )
    
    blocks[0].add_gradient(
            parameter="positions",
            data=forces,
            samples=samples_gradients,
            components=[gradients_labels]
            )

    return TensorMap(Labels.single(), blocks)
