"""Paths through the sample, for absorption: how far a ray travels inside it from a voxel to its edge.

The sample is a map of the fraction of each voxel it fills, in a 2D slice (reconstruction order, see
:mod:`anri.geom.scan`), taken to be the same above and below the slice (a prism). Then the path from a voxel out of
the sample along any direction depends only on the voxel and the direction's angle in the slice plane, phi: the path
in the plane is :func:`edge_distances` ``D[voxel, phi]``, and a ray leaving the plane at an elevation goes
``D / cos(elevation)`` (:func:`path_lengths`). One table serves the incoming beam (looking back towards the source)
and every diffracted ray, for every prediction.

Lengths are in the units of the voxel positions (micrometres in anri); the attenuation coefficient is in the inverse
unit (:func:`anri.crystal.structure.attenuation`).
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jax.typing import ArrayLike

from anri.maths.rotation import rot_x, rot_y, rot_z


@partial(jax.jit, static_argnames=("n_phi",))
def edge_distances(sample_map: jax.Array, step: float, pos: jax.Array, n_phi: int = 360) -> jax.Array:
    """Path inside the sample from each position to the sample's edge, along each direction in the slice plane.

    For each direction ``phi_k = 2 pi k / n_phi`` (``(cos phi, sin phi, 0)`` in the sample frame), the map is resampled
    on a grid aligned with it, summed backwards along it, and that is sampled at the positions (bilinear, so
    boundaries are resolved to about a voxel).

    Parameters
    ----------
    sample_map
        [n, n] fraction of each voxel inside the sample (0 to 1), reconstruction order, centred on the rotation axis
        (:func:`anri.geom.scan.recon_positions`)
    step
        Voxel size
    pos
        [Nv, 2 or 3] sample-frame positions (x, y)
    n_phi
        Directions

    Returns
    -------
    jax.Array
        [Nv, n_phi] path inside the sample, in the units of step
    """
    n = sample_map.shape[0]
    c0 = n // 2  # recon index of the rotation axis
    h = math.ceil(n * 0.7072) + 2  # half-size of a grid covering the map in any direction
    t = jnp.arange(-h, h + 1, dtype=jnp.float32)  # along the direction (voxels)
    s = t  # across it
    T, S = jnp.meshgrid(t, s, indexing="ij")
    px, py = pos[:, 0] / step, pos[:, 1] / step

    def one(phi: jax.Array) -> jax.Array:
        u = jnp.array([jnp.cos(phi), jnp.sin(phi)])
        x, y = T * u[0] - S * u[1], T * u[1] + S * u[0]  # sample frame (voxels)
        m = map_coordinates(sample_map, [x + c0, c0 - y], order=1, cval=0.0)  # recon: ri = x + n//2, rj = n//2 - y
        d = (jnp.cumsum(m[::-1], axis=0)[::-1] - 0.5 * m) * step  # from each point onwards, along +t
        tp, sp = px * u[0] + py * u[1], -px * u[1] + py * u[0]
        return map_coordinates(d, [tp + h, sp + h], order=1, cval=0.0)

    phis = jnp.arange(n_phi, dtype=jnp.float32) * (2 * jnp.pi / n_phi)
    return jax.lax.map(one, phis).T


def ray_directions(
    k_in_lab: ArrayLike, k_out_lab: ArrayLike, omega: ArrayLike, wedge: ArrayLike = 0.0, chi: ArrayLike = 0.0
) -> tuple[jax.Array, jax.Array]:
    """Return the unit directions, in the sample frame, of the two paths from a diffracting voxel to the sample's edge.

    Parameters
    ----------
    k_in_lab
        [3] incoming beam direction in the lab frame
    k_out_lab
        [..., 3] diffracted directions in the lab frame (e.g. :func:`anri.diffract.scatter.tth_eta_to_k_out`)
    omega
        [...] omega of each (degrees)
    wedge, chi
        Goniometer tilts (degrees), as :func:`anri.geom.gonio.lab_to_sample`

    Returns
    -------
    d_in, d_out: jax.Array
        [..., 3] each: back along the incoming beam (towards the source), and along the diffracted ray
    """
    om = jnp.asarray(omega)
    M = jax.vmap(rot_z)(om.ravel()).reshape(*om.shape, 3, 3)  # sample -> lab is W C R, so lab -> sample is R^T C^T W^T
    WC = rot_y(wedge) @ rot_x(chi)

    def to_sample(v: jax.Array) -> jax.Array:
        v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        return jnp.einsum("...ji,...j->...i", M, jnp.einsum("ji,...j->...i", WC, v))

    k_in = jnp.broadcast_to(jnp.asarray(k_in_lab, jnp.float32), om.shape + (3,))
    return to_sample(-k_in), to_sample(jnp.asarray(k_out_lab, jnp.float32))


def path_lengths(D: jax.Array, d: jax.Array, min_horizontal: float = 0.05) -> jax.Array:
    """Path inside the sample from voxels along directions, from their :func:`edge_distances`.

    Parameters
    ----------
    D
        [Nv, n_phi] edge distances of the voxels
    d
        [Nv, ..., 3] unit directions in the sample frame
    min_horizontal
        Smallest in-plane fraction of a direction used (a nearly vertical ray would leave a real sample through its
        top or bottom, not travel ``D / cos`` through an endless prism)

    Returns
    -------
    jax.Array
        [Nv, ...] path lengths
    """
    n_phi = D.shape[-1]
    horizontal = jnp.maximum(jnp.hypot(d[..., 0], d[..., 1]), min_horizontal)
    f = (jnp.arctan2(d[..., 1], d[..., 0]) % (2 * jnp.pi)) * (n_phi / (2 * jnp.pi))
    k0 = jnp.floor(f).astype(jnp.int32)
    w = f - k0
    Dv = D.reshape(D.shape[0], *([1] * (d.ndim - 2)), n_phi)
    a = jnp.take_along_axis(Dv, (k0 % n_phi)[..., None], axis=-1)[..., 0]
    b = jnp.take_along_axis(Dv, ((k0 + 1) % n_phi)[..., None], axis=-1)[..., 0]
    return ((1 - w) * a + w * b) / horizontal


def transmission(D: jax.Array, d_in: jax.Array, d_out: jax.Array, mu: float) -> jax.Array:
    """Fraction of the intensity that survives absorption along the incoming and the diffracted paths.

    Parameters
    ----------
    D
        [Nv, n_phi] edge distances (:func:`edge_distances`)
    d_in, d_out
        [Nv, ..., 3] directions from :func:`ray_directions`
    mu
        Linear attenuation coefficient, in the inverse units of D

    Returns
    -------
    jax.Array
        [Nv, ...] exp(-mu (L_in + L_out))
    """
    return jnp.exp(-mu * (path_lengths(D, d_in) + path_lengths(D, d_out)))
