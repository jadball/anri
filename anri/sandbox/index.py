import jax.numpy as jnp
import jax
from jax import lax, jit, vmap

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", 'cpu')


# very basic utilites
@jit
def unit(v):
    """Normalize a single vector"""
    return v / jnp.linalg.norm(v)

unit_many = jit(vmap(unit))

@jit
def _g_to_ds(gvec):
    # compute d-star from g-vector
    # it's just the length
    ds = jnp.linalg.norm(gvec)
    return ds

g_to_ds = jit(vmap(_g_to_ds))

@jit
def assign_peaks_to_rings(ds, ringds, ds_tol):
    # (npks, nrings)
    dists = jnp.abs(ds[:, None] - ringds[None, :])

    # index of closest ring (npks,)
    ring_assignments = jnp.argmin(dists, axis=1)

    # gather distances to assigned rings
    min_dists = jnp.take_along_axis(dists, ring_assignments[:, None], axis=1)[:, 0]

    # apply tolerance mask
    return jnp.where(min_dists < ds_tol, ring_assignments, -1)


@jit
def anglehkls(hkl1, hkl2, rmt):
    """
    Compute angles between all pairs of reciprocal lattice vectors in hkl1 and hkl2.
    
    hkl1: (N,3) array
    hkl2: (M,3) array
    rmt: (3,3) reciprocal metric tensor
    Returns: 
        angles: (N,M) array of angles in degrees
        cosines: (N,M) array of cosines
    """
    # Compute squared norms under metric
    g1 = jnp.einsum('ni,ij,nj->n', hkl1, rmt, hkl1)  # (N,)
    g2 = jnp.einsum('mi,ij,mj->m', hkl2, rmt, hkl2)  # (M,)
    
    # Compute dot products under metric
    g12 = jnp.einsum('ni,ij,mj->nm', hkl1, rmt, hkl2)  # (N,M)
    
    # Compute cosines
    cos_theta = g12 / jnp.sqrt(g1[:,None] * g2[None,:])
    
    return cos_theta


@jax.jit
def build_frame_and_apply(A, v1, v2):
    """
    Build orthonormal frame from v1, v2 and apply matrix A.

    A: (3,3) matrix to multiply by
    v1, v2: input 3-vectors
    Returns: A @ [u1, u2, u3] (3x3)
    """
    u1 = unit(v1)
    u3 = unit(jnp.cross(v1, v2))
    u2 = jnp.cross(u1, u3)
    M = jnp.stack([u1, u2, u3], axis=1)
    return A @ M


# BTmat wrapper
@jax.jit
def BTmat(h1, h2, B, BI):
    g1 = B @ h1
    g2 = B @ h2
    return build_frame_and_apply(BI, g1, g2)


@jax.jit
def quickorient(ubi_sq, bt_sq):
    """
    ubi_sq: jnp array of shape (3,3) 
    bt_sq: jnp array of shape (3,3)
    """
    UBI = ubi_sq.ravel()
    BT = bt_sq.ravel()
    
    # Compute g1 x g2
    M6 = UBI[1] * UBI[5] - UBI[2] * UBI[4]
    M7 = UBI[2] * UBI[3] - UBI[0] * UBI[5]
    M8 = UBI[0] * UBI[4] - UBI[1] * UBI[3]

    # Normalize g0
    t0 = jnp.sqrt(UBI[0]**2 + UBI[1]**2 + UBI[2]**2)
    M0 = UBI[0] / t0
    M1 = UBI[1] / t0
    M2 = UBI[2] / t0

    # Normalize g1 x g2
    t1 = jnp.sqrt(M6**2 + M7**2 + M8**2)
    M6 /= t1
    M7 /= t1
    M8 /= t1

    # Compute u2 = u1 x u3
    M3 = M1 * M8 - M2 * M7
    M4 = M2 * M6 - M0 * M8
    M5 = M0 * M7 - M1 * M6

    # Compute UBI = BT @ M (manual dot)
    UBI_new = jnp.array([
        BT[0] * M0 + BT[1] * M3 + BT[2] * M6,
        BT[0] * M1 + BT[1] * M4 + BT[2] * M7,
        BT[0] * M2 + BT[1] * M5 + BT[2] * M8,
        BT[3] * M0 + BT[4] * M3 + BT[5] * M6,
        BT[3] * M1 + BT[4] * M4 + BT[5] * M7,
        BT[3] * M2 + BT[4] * M5 + BT[5] * M8,
        BT[6] * M0 + BT[7] * M3 + BT[8] * M6,
        BT[6] * M1 + BT[7] * M4 + BT[8] * M7,
        BT[6] * M2 + BT[7] * M5 + BT[8] * M8
    ])
    
    return UBI_new.reshape(3,3)


@jax.jit
def score(ubi, gv, tol):
    """
    Count the number of g-vectors indexed by UBI within tolerance.

    ubi: (3,3) orientation matrix
    gv: (N,3) array of g-vectors
    tol: scalar tolerance
    Returns: integer number of peaks within tolerance
    """
    # Transform gv by ubi
    hkl = gv @ ubi.T  # shape (N,3)

    # Compute distance to nearest integer
    dhkl = hkl - jnp.round(hkl)

    # Squared distance
    drlv2 = jnp.sum(dhkl**2, axis=1)

    # Count peaks where drlv2 < tol^2
    n = jnp.sum(drlv2 < tol**2)
    return n


@jax.jit
def assign_hits(
    gv: jnp.ndarray,     # (N,3)
    ra: jnp.ndarray,     # (N,)
    ring1: int,
    ring2: int,
    hkls1: jnp.ndarray,  # (M,3)
    hkls2: jnp.ndarray,  # (K,3)
    rmt,
    tol: float = 1e-5,

):

    N = gv.shape[0]

    coses = anglehkls(hkls1, hkls2, rmt).reshape(-1)

    # Mask ±1
    bad = (jnp.abs(coses - 1.0) < 1e-5) | (jnp.abs(coses + 1.0) < 1e-5)
    coses = jnp.where(bad, jnp.nan, coses)     # shape stays M*K

    # Normalize gv                
    # gv_n = normalize(gv)  # (N,3)
    gv_n = unit_many(gv)
    
    # Masks for ring membership
    mask1 = (ra == ring1)                      # (N,)
    mask2 = (ra == ring2)                      # (N,)

    # Mask out non-ring entries with NaN
    n1 = jnp.where(mask1[:, None], gv_n, jnp.nan) # (N,3)
    n2 = jnp.where(mask2[:, None], gv_n, jnp.nan) # (N,3)

    # Pairwise dot products (N,N) with NaNs where invalid
    costheta = n1 @ n2.T                       # (N,N)

    # |costheta - theoretical|  → (N,N,M*K)
    diff_matrix = jnp.abs(costheta[:, :, None] - coses[None, None, :])

    # Best match difference for each (i,j)
    best_diff_ij = jnp.nanmin(diff_matrix, axis=2)  # (N,N)

    # Best j per i
    best_j = jnp.nanargmin(best_diff_ij, axis=1)    # (N,)
    best_diff = jnp.nanmin(best_diff_ij, axis=1)    # (N,)

    # Hit mask: ring1 & match tolerance
    hit_mask = mask1 & (best_diff < tol)            # (N,)

    # Return **full-length arrays**, no shrinking
    return {
        "diffs": best_diff.astype(jnp.float32),     # (N,)
        "i1": jnp.arange(N, dtype=jnp.int32),       # (N,)
        "i2": best_j.astype(jnp.int32),             # (N,)
        "mask": hit_mask.astype(jnp.bool_),         # (N,)
        "coses": coses
    }