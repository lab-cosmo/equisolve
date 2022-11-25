import numpy as np
import ase

from equistore import TensorMap, Labels
from equistore.block import TensorBlock, Gradient


# needs to be defined globally to ensure that energy_to_tensormap and property_to_tensormap use the same function
def _get_default_positions_gradient_extraction(frame, positions_gradient):
    if isinstance(frame, ase.Atoms):
        return lambda frame : frame.arrays[positions_gradient]
    elif isinstance(frames, chemfiles.Frame):
        raise NotImplementedError("Extracting positions gradient out of frame objects of type chemfiles.Frame has not been implemented yet.")
    else:
        raise NotImplementedError(f"Extracting positions gradient out of frame objects of type {type(frame)} is not supported.")


def energy_to_tensormap(frames, energy=None, forces=None, stress=None):
    if not isinstance(frames, list):
        frames = [frames]

    if isinstance(forces, type(lambda x : x)):
        get_forces = lambda frame : -forces(frame)
    elif isinstance(forces, str):
        default_positions_gradient_extraction = _get_default_positions_gradient_extraction(frames[0], forces)
        get_forces = lambda frame : -default_positions_gradient_extraction(frame)

    return property_to_tensormap(frames, energy, get_forces, stress)


def property_to_tensormap(frames, property=None, positions_gradient=None, cell_gradient=None, is_structure_property=True):
    """
    Creates TensorMap out of frame(s) containing perties

    :param frames: frame or list of frames
        Must be an object or a list of objects that contain the properties.
        Currenly supported are ase.Atoms objects.

    :param property: string or function that returns property
        The string is used to extract the property out of the frame.
        The way the string is used depends on the frame type:

        ase.Atoms
        ```python
            frame_property = frame.info[property]
        ```

        If a function is given, the function is used to extract the property out
        of the frame
        ```python
            frame_property = property(frame)
        ```
        chemfiles.Frame
        TODO

    :param positions_gradient: string or function that returns array of shape (len(frame), 3)
        The string is used to extract the gradient of the property wrt. positions out of the frame.
        The way the string is used depends on the frame type:

        ase.Atoms
        ```python
            property_positions_gradient = frame.arrays[positions_gradient]
        ```
        chemfiles.Frame
        TODO

        If a function is given, the function is used to extract the the gradient of the property wrt. positions out
        of the frame
        ```python
            property_positions_gradient = positions_gradient(frame)
        ```

    :param cell_gradient: string or function that returns array of shape (3, 3)
        NOT IMPLEMENTED

    """

    if not isinstance(frames, list):
        frames = [frames]
    if isinstance(frames[0], ase.Atoms):
        return _ase_frames_property_to_tensormap(frames, property, positions_gradient, cell_gradient, is_structure_property)
    elif isinstance(frames[0], chemfiles.Frame):
        raise NotImplementedError("Extracting porperties out of frame objects of type chemfiles.Frame has not been implemented yet.")
    else:
        raise NotImplementedError(f"Extracting porperties out of frame objects of type {type(frames[0])} is not supported.")


def _ase_frames_property_to_tensormap(frames, property=None, positions_gradient=None, cell_gradient=None, is_structure_property=True):
    if cell_gradient is not None:
        raise NotImplementedError("cell_gradient is not supported at the moment.")
    if not(is_structure_property):
        raise NotImplementedError("Support for environment properties has not been implemented yet.")

    if isinstance(property, str):
        get_property = lambda frame : frame.info[property]
    else:
        get_property = property
    if isinstance(positions_gradient, str):
        get_positions_gradient = _get_default_positions_gradient_extraction(frames[0], positions_gradient)
    else:
        get_positions_gradient = positions_gradient
    #if isinstance(cell_gradient, str):
    #    get_stress = lambda frame : -frame.arrays[property]


    keys = Labels(
        names=["idx"],
        values=np.array([[0]], dtype=np.int32)
    )
    #keys = Labels.single()

    if property is None:
        raise NotImplementedError("Needs to be implemented before PR gets merged.")
    else:
        property_values = np.array([get_property(frame) for frame in frames])
        # -1 dimension because of property dimension

        samples_property = Labels(
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
            values=property_values.reshape(-1,1),
            samples=samples_property,
            components=[],
            properties=Labels(['property'], np.array([(0,)], dtype=np.int32))
        )]

    if positions_gradient is not None:
        positions_gradient  = np.concatenate([get_positions_gradient(frame) for frame in frames], axis=0).reshape(-1, 3, 1)
        positions_gradient_labels = Labels(
            ['direction'],
            np.array([(0,), (1,), (2,)], dtype=np.int32)
        )

        blocks[0].add_gradient(
                parameter="positions",
                data=positions_gradient,
                samples=samples_gradients,
                components=[positions_gradient_labels]
                )

    return TensorMap(Labels.single(), blocks)
