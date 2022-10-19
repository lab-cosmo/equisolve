import numpy as np
from equistore import Labels, TensorBlock, TensorMap


def structure_sum(descriptor):
    blocks = []
    for _, block in descriptor:

        # no lambda kernels for now
        assert len(block.components) == 0

        structures = np.unique(block.samples["structure"])
        properties = block.properties

        result = np.zeros_like(
            block.values,
            shape=(len(structures), properties.shape[0]),
        )

        if block.has_gradient("positions"):
            do_gradients = True
            gradient = block.gradient("positions")

            assert np.all(np.unique(gradient.samples["structure"]) == structures)

            gradient_data = []
            new_gradient_samples = []
            atom_index_positions = []
            for structure_i, s1 in enumerate(structures):
                mask = gradient.samples["structure"] == s1
                atoms = np.unique(gradient.samples[mask]["atom"])

                gradient_data.append(
                    np.zeros_like(
                        gradient.data,
                        shape=(len(atoms), 3, properties.shape[0]),
                    )
                )

                new_gradient_samples.append(
                    np.array(
                        [[structure_i, s1, atom] for atom in atoms],
                        dtype=np.int32,
                    )
                )

                atom_index_positions.append({atom: i for i, atom in enumerate(atoms)})

            new_gradient_samples = Labels(
                names=["sample", "structure", "atom"],
                values=np.concatenate(new_gradient_samples),
            )

        else:
            do_gradients = False

        for structure_i, s1 in enumerate(structures):
            s1_idx = block.samples["structure"] == s1

            result[structure_i, :] = np.sum(block.values[s1_idx, :], axis=0)

            if do_gradients:
                for sample_i in np.where(gradient.samples["structure"] == s1)[0]:
                    grad_sample = gradient.samples[sample_i]

                    atom_i = atom_index_positions[structure_i][grad_sample["atom"]]
                    gradient_data[structure_i][atom_i, :, :] += gradient.data[
                        sample_i, :, :
                    ]

        new_block = TensorBlock(
            values=result,
            samples=Labels(["structure"], structures.reshape(-1, 1)),
            components=[],
            properties=properties,
        )

        if do_gradients:
            gradient_data = np.vstack(gradient_data).reshape(-1, 3, properties.shape[0])
            new_block.add_gradient(
                "positions", gradient_data, new_gradient_samples, gradient.components
            )

        blocks.append(new_block)

    return TensorMap(keys=descriptor.keys, blocks=blocks)
