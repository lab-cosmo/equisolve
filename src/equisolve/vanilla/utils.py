# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

# TODO: This whole thing should live in equistore directly

import numpy as np
from equistore import Labels, TensorBlock, TensorMap


def normalize(descriptor):
    """Normalize"""
    blocks = []
    for _, block in descriptor:
        # only deal with invariants for now
        assert len(block.components) == 0
        assert len(block.values.shape) == 2

        norm = np.linalg.norm(block.values, axis=1)
        normalized_values = block.values / norm[:, None]

        new_block = TensorBlock(
            values=normalized_values,
            samples=block.samples,
            components=[],
            properties=block.properties,
        )

        if block.has_gradient("positions"):
            gradient = block.gradient("positions")

            gradient_data = gradient.data / norm[gradient.samples["sample"], None, None]

            # gradient of x_i = X_i / N_i is given by
            # 1 / N_i \grad X_i - x_i [x_i @ 1 / N_i \grad X_i]
            for sample_i, (sample, _, _) in enumerate(gradient.samples):
                dot = gradient_data[sample_i] @ normalized_values[sample].T

                gradient_data[sample_i, 0, :] -= dot[0] * normalized_values[sample, :]
                gradient_data[sample_i, 1, :] -= dot[1] * normalized_values[sample, :]
                gradient_data[sample_i, 2, :] -= dot[2] * normalized_values[sample, :]

            new_block.add_gradient(
                "positions", gradient_data, gradient.samples, gradient.components
            )

        blocks.append(new_block)

    return TensorMap(descriptor.keys, blocks)


def dot(lhs_descriptor, rhs_descriptor, do_normalize=True):
    assert len(lhs_descriptor.keys) == len(rhs_descriptor.keys)
    if len(lhs_descriptor.keys) != 0:
        assert np.all(lhs_descriptor.keys == rhs_descriptor.keys)

    if do_normalize:
        lhs_descriptor = normalize(lhs_descriptor)

    blocks = []
    for key, lhs in lhs_descriptor:
        rhs = rhs_descriptor.block(key)
        assert np.all(lhs.properties == rhs.properties)

        # only deal with invariants for now
        assert len(lhs.components) == 0
        assert len(rhs.components) == 0

        samples = lhs.samples
        properties = rhs.samples

        block = TensorBlock(
            values=lhs.values @ rhs.values.T,
            samples=samples,
            components=[],
            properties=properties,
        )

        if lhs.has_gradient("positions"):
            gradient = lhs.gradient("positions")

            gradient_data = gradient.data @ rhs.values.T

            block.add_gradient(
                "positions",
                gradient_data,
                gradient.samples,
                gradient.components,
            )

        if rhs.has_gradient("positions"):
            print("ignoring gradients of kernel support points")

        blocks.append(block)

    return TensorMap(lhs_descriptor.keys, blocks)


def power(descriptor, zeta):
    assert zeta >= 1

    blocks = []
    for _, block in descriptor:
        new_block = TensorBlock(
            np.float_power(block.values, zeta),
            block.samples,
            block.components,
            block.properties,
        )

        if block.has_gradient("positions"):
            gradient = block.gradient("positions")

            if zeta > 1:
                values_pow_zeta_m_1 = zeta * np.float_power(
                    block.values[gradient.samples["sample"], :], zeta - 1
                )
                gradient_data = gradient.data * values_pow_zeta_m_1[:, None, :]
            else:
                assert zeta == 1
                gradient_data = gradient.data

            new_block.add_gradient(
                "positions",
                gradient_data,
                gradient.samples,
                gradient.components,
            )

        blocks.append(new_block)

    return TensorMap(descriptor.keys, blocks)


def structure_sum(descriptor, sum_properties=False):
    blocks = []
    for _, block in descriptor:

        # no lambda kernels for now
        assert len(block.components) == 0

        structures = np.unique(block.samples["structure"])

        if sum_properties:
            ref_structures = np.unique(block.properties["structure"])
            properties = Labels(["structure"], ref_structures.reshape(-1, 1))
        else:
            properties = block.properties

        result = np.zeros_like(block.values, (len(structures), properties.shape[0]))

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
                        (len(atoms), 3, properties.shape[0]),
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

        if sum_properties:
            for structure_i, s1 in enumerate(structures):
                s1_idx = block.samples["structure"] == s1

                for structure_j, s2 in enumerate(ref_structures):
                    s2_idx = block.properties["structure"] == s2
                    result[structure_i, structure_j] = np.sum(
                        block.values[s1_idx, :][:, s2_idx]
                    )

                    if do_gradients:
                        idx = np.where(gradient.samples["structure"] == s1)[0]
                        for sample_i in idx:
                            grad_sample = gradient.samples[sample_i]
                            atom_i = atom_index_positions[structure_i][
                                grad_sample["atom"]
                            ]

                            term = np.sum(
                                gradient.data[sample_i, :, :][:, s2_idx], axis=1
                            )

                            gradient_data[structure_i][atom_i, :, structure_j] += term
        else:
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


def detach(descriptor):
    if isinstance(descriptor.block(0).values, torch.Tensor):
        blocks = []
        for _, block in descriptor:
            blocks.append(
                TensorBlock(
                    values=block.values.detach(),
                    samples=block.samples,
                    components=block.components,
                    properties=block.properties,
                )
            )
        descriptor = TensorMap(descriptor.keys, blocks)

    return descriptor
