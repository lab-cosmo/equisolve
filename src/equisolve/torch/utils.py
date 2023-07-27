from typing import List, Tuple

from equistore import TensorMap, join


def _equistore_collate_fn(data: List[Tuple[TensorMap, ...]]) -> List[TensorMap]:
    """
    collates a list of tuples of TensorMaps

    data: is a list of tuples with an arbitrary number of TensorMaps,
    each tuple contains the same number of TensorMaps
    """

    data = [list(x) for x in zip(*data)]
    joined_maps = [join(dat, axis="samples") for dat in data]

    return joined_maps
