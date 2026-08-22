"""Custom unbatched Lorem variant for benchmarking PME and Ewald."""

import jax
import jax.numpy as jnp
import numpy as np

import e3x
import flax.linen as nn
from jaxpme import PME, Ewald

from lorem.models.backbone import (
    MLP,
    Initial,
    RadialCoefficients,
    Update,
    degree_wise_repeat_last_axis,
    spherical_norm_last_axis,
)
from lorem.models.backbone import (
    _masked as masked,
)


class LoremUnbatched(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 6
    max_degree_lr: int = 2
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 8
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"
    num_message_passing: int = 0
    equivariant_message_passing: bool = True
    initialize_node_features: bool = True
    lr_calculator: str = "pme"  # "pme" or "ewald"
    lr: bool = True

    @nn.compact
    def __call__(
        self,
        Z_i,
        positions,
        cell,
        centers,
        others,
        cell_shifts,
        k_grid,
        smearing,
        pbc,
    ):
        i = centers
        j = others

        R_ij = (
            positions[j]
            - positions[i]
            + jnp.einsum("pA,Aa->pa", cell_shifts, cell)
        )

        num_atoms = Z_i.shape[0]
        num_pairs = R_ij.shape[0]

        atom_mask = jnp.ones(num_atoms, dtype=bool)
        pair_mask = jnp.ones(num_pairs, dtype=bool)

        max_degree = self.max_degree
        max_degree_lr = self.max_degree_lr
        num_l = self.max_degree + 1
        num_lm = int((self.max_degree + 1) ** 2)

        d = self.num_features
        s = self.num_spherical_features

        l_factors = (
            jnp.array(
                [(2 * l + 1) for l in range(max_degree + 1)], dtype=float
            )
            ** 0.25
        )

        # -- initial embeddings --
        radial, spherical, species, cutoffs, r_ij = Initial(
            cutoff=self.cutoff,
            max_degree=self.max_degree,
            num_features=self.num_features,
            num_radial=self.num_radial,
            num_species=self.num_species,
            num_spherical_features=self.num_spherical_features,
            cutoff_fn=self.cutoff_fn,
            radial_basis=self.radial_basis,
        )(
            R_ij,
            Z_i,
            pair_mask,
            atom_mask,
        )

        # -- learned linear transformation of radial expansion --
        edges_scalar = RadialCoefficients(d)(
            jnp.concatenate([species[i], species[j]], axis=-1),
            radial,
            cutoffs,
            pair_mask,
        )

        # -- initial scalar and equivariant (spherical) node features
        if self.initialize_node_features:
            nodes_scalar = masked(
                nn.Dense(d, use_bias=True), species, atom_mask
            )
        else:
            nodes_scalar = jnp.zeros(
                (num_atoms, d), dtype=species.dtype
            )

        updates = (
            jax.ops.segment_sum(
                masked(
                    nn.Dense(d, use_bias=False), edges_scalar, pair_mask
                ),
                i,
                num_segments=num_atoms,
            )
            * atom_mask[..., None]
        )
        nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

        coefficients = masked(
            nn.Dense(num_l * s, use_bias=False), edges_scalar, pair_mask
        ).reshape(num_pairs, num_l, s)
        coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
        edges_spherical = jnp.einsum(
            "plf,pl->plf", coefficients, spherical
        )

        nodes_spherical = (
            jax.ops.segment_sum(
                edges_spherical.reshape(num_pairs, 1, num_lm, s),
                i,
                num_segments=num_atoms,
            )
            * atom_mask[..., None, None, None]
        )
        nodes_spherical = e3x.nn.TensorDense(
            use_bias=False, include_pseudotensors=False
        )(nodes_spherical)

        # -- mix equivariant information into scalar node features --
        norms = spherical_norm_last_axis(nodes_spherical, max_degree)
        updates = (norms * l_factors[None, None, :, None]).reshape(
            num_atoms, -1
        )

        nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

        # -- initial prediction --
        energy = masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[
            ..., 0
        ]

        # -- message passing (if turned on) --
        for _ in range(self.num_message_passing):
            edges_scalar = RadialCoefficients(d)(
                jnp.concatenate(
                    [nodes_scalar[i], nodes_scalar[j]], axis=-1
                ),
                radial,
                cutoffs,
                pair_mask,
            )
            updates = (
                jax.ops.segment_sum(
                    masked(
                        nn.Dense(d, use_bias=False),
                        edges_scalar,
                        pair_mask,
                    ),
                    i,
                    num_segments=num_atoms,
                )
                * atom_mask[..., None]
            )
            nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            if self.equivariant_message_passing:
                coefficients = masked(
                    nn.Dense(num_l * s, use_bias=False),
                    edges_scalar,
                    pair_mask,
                ).reshape(num_pairs, num_l, s)
                coefficients = degree_wise_repeat_last_axis(
                    coefficients, max_degree
                )
                edges_spherical = jnp.einsum(
                    "plf,pl->plf", coefficients, spherical
                ).reshape(num_pairs, 1, num_lm, s)

                messages = (
                    e3x.nn.MessagePass(include_pseudotensors=False)(
                        nodes_spherical,
                        edges_spherical,
                        dst_idx=i,
                        src_idx=j,
                    )
                    * atom_mask[..., None, None, None]
                )
                nodes_spherical = e3x.nn.Tensor(
                    include_pseudotensors=False
                )(
                    e3x.nn.Dense(use_bias=False, features=s)(
                        nodes_spherical
                    ),
                    e3x.nn.Dense(use_bias=False, features=s)(messages),
                )

                norms = spherical_norm_last_axis(
                    nodes_spherical, max_degree
                )
                updates = (
                    norms * l_factors[None, None, :, None]
                ).reshape(num_atoms, -1)
                nodes_scalar = Update(d)(
                    nodes_scalar, updates, atom_mask
                )

            # -- residual prediction --
            energy += masked(
                MLP(features=[d, d, 1]), nodes_scalar, atom_mask
            )[..., 0]

        if self.lr:
            # -- compute LR potentials via PME --
            scalar_charges = masked(
                MLP(features=[2 * d, 1]), nodes_scalar, atom_mask
            )
            spherical_charges = e3x.nn.TensorDense(
                features=1,
                use_bias=False,
                max_degree=max_degree_lr,
                include_pseudotensors=False,
            )(nodes_spherical).reshape(num_atoms, -1)
            charges = jnp.concatenate(
                [scalar_charges, spherical_charges], axis=-1
            )

            def _potentials(
                charges, cell, positions, i, j, cell_shifts,
                k_grid, smearing, pbc,
            ):
                calculator = (
                    PME() if self.lr_calculator == "pme" else Ewald()
                )
                return jax.vmap(
                    lambda q: calculator.potentials(
                        q, cell, positions, i, j, cell_shifts,
                        k_grid, smearing, pbc=pbc,
                    ),
                    in_axes=-1,
                    out_axes=-1,
                )(charges)

            potentials = _potentials(
                charges, cell, positions, i, j, cell_shifts,
                k_grid, smearing, pbc,
            )

            scalar_potential = potentials[..., 0][..., None]
            spherical_potential = potentials[..., 1:].reshape(
                num_atoms, 1, -1, 1
            )

            # -- combine LR potentials back into local features --
            spherical_potential = e3x.nn.Dense(s, use_bias=False)(
                spherical_potential
            )
            spherical_updates = e3x.nn.Tensor(
                include_pseudotensors=False
            )(spherical_potential, nodes_spherical)

            norms = spherical_norm_last_axis(
                spherical_updates, max_degree
            )
            norms = (norms * l_factors[None, None, :, None]).reshape(
                num_atoms, -1
            )
            updates = jnp.concatenate(
                [scalar_potential, norms], axis=-1
            )
            nodes_scalar = Update(d)(
                nodes_scalar, updates, atom_mask
            )

            # -- residual prediction --
            energy += masked(
                MLP(features=[d, d, 1]), nodes_scalar, atom_mask
            )[..., 0]

        return energy


def next_smooth(n):
    """Smallest m >= n such that m = 2^a * 3^b * 5^c * 7^d."""
    n = int(n)
    if n <= 1:
        return 1
    # brute force: check each candidate
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def get_kgrid_smooth(cell, mesh_spacing):
    """Like get_kgrid_mesh but pads to next {2,3,5,7}-smooth number."""
    from jaxpme.kspace import get_kgrid_ewald

    start = np.array(get_kgrid_ewald(cell, mesh_spacing).shape)
    actual = 2 * start + 1
    ns = tuple(next_smooth(int(n)) for n in actual)

    return jnp.ones(ns)


def prepare_inputs(atoms, cutoff, mesh_spacing, smearing, mode="pme"):
    """Convert ASE atoms to plain arrays for unbatched Lorem."""
    from vesin import ase_neighbor_list as neighbor_list
    from jaxpme.kspace import get_kgrid_ewald

    Z_i = jnp.array(atoms.get_atomic_numbers())
    positions = jnp.array(atoms.get_positions(), dtype=jnp.float32)
    cell = jnp.array(atoms.get_cell().array, dtype=jnp.float32)
    pbc = jnp.array(atoms.get_pbc())

    i, j, S = neighbor_list("ijS", atoms, cutoff)
    centers = jnp.array(i)
    others = jnp.array(j)
    cell_shifts = jnp.array(S, dtype=jnp.float32)

    if mode == "pme":
        k_grid = get_kgrid_smooth(cell, mesh_spacing)
    else:
        k_grid = get_kgrid_ewald(cell, mesh_spacing)

    return (
        Z_i, positions, cell, centers, others, cell_shifts,
        k_grid, smearing, pbc,
    )


# backward compat
LoremPME = LoremUnbatched


def predict(model, params, inputs):
    """Energy + forces prediction for LoremPME."""
    Z_i, positions, cell, centers, others, cell_shifts, \
        k_grid, smearing, pbc = inputs

    def energy_fn(positions):
        energies = model.apply(
            params, Z_i, positions, cell, centers, others,
            cell_shifts, k_grid, smearing, pbc,
        )
        return jnp.sum(energies)

    energy, pos_grad = jax.value_and_grad(energy_fn)(positions)
    return {"energy": energy, "forces": -pos_grad}
