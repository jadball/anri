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


# @jax.jit
# def build_frame_and_apply(v1, v2, A):
#     """
#     Build orthonormal frame from v1, v2 and apply matrix A.

#     A: (3,3) matrix to multiply by
#     v1, v2: input 3-vectors
#     Returns: A @ [u1, u2, u3] (3x3)
#     """
#     g3 = jnp.cross(v1, v2)
#     u1 = unit(v1)
#     u3 = unit(g3)
#     u2 = jnp.cross(u1, u3)
#     # M = jnp.stack([u1, u2, u3], axis=1)
#     M = jnp.array((u1, u2, u3))
#     return A @ M


@jax.jit
def get_rotmat(v1, v2):
    """
    Build orthonormal frame from v1, v2

    v1, v2: input 3-vectors
    Returns: [u1, u2, u3] (3x3)
    """
    g3 = jnp.cross(v1, v2)
    u1 = unit(v1)
    u3 = unit(g3)
    u2 = jnp.cross(u1, u3)
    M = jnp.array((u1, u2, u3))
    return M


@jax.jit
def BTmat(h1, h2, B, BI):
    g1 = B @ h1
    g2 = B @ h2
    M = get_rotmat(g1, g2)
    return BI @ M.T


@jax.jit
def quickorient(g1, g2, BT):
    M = get_rotmat(g1, g2)
    return BT @ M


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


@jax.jit
def find(parents, x):
    # Path compression
    def loop_body(val):
        px = val
        return parents[px]
    
    def cond(val):
        return parents[val] != val

    return jax.lax.while_loop(cond, loop_body, x)

@jax.jit
def union(parents, x, y):
    px = find(parents, x)
    py = find(parents, y)
    parents = parents.at[px].set(py)
    return parents

@jax.jit
def connected_components(adj):
    n = adj.shape[0]
    parents = jnp.arange(n)

    def body_fun(i, parents):
        def inner_body(j, parents):
            parents = jax.lax.cond(adj[i, j] == 1,
                               lambda p: union(p, i, j),
                               lambda p: p,
                               parents)
            return parents
        parents = jax.lax.fori_loop(0, n, inner_body, parents)
        return parents

    parents = jax.lax.fori_loop(0, n, body_fun, parents)
    
    # Flatten parents
    def compress_parent(i, parents):
        return parents.at[i].set(find(parents, i))
    parents = jax.lax.fori_loop(0, n, compress_parent, parents)
    
    return jnp.unique(parents, size=n, fill_value=-1)

HKL0 = jnp.array([[0, 0, 1, 1, -1, 1, -1, 0, 0, 1, -1, 1, 1, 3, 11],
                 [0, 1, 0, 1, 1, 0, 0, 1, -1, 1, 1, -1, 1, 2, 12],
                 [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 1, 13]],)  # first unit cell

@jax.jit
def test_combo(ga, gb, BT, gt):
    # (ga, gb) from the left
    # BT from the right
    # gt from the left
    UBI = quickorient(ga, gb, BT)
    npk = score(UBI, gt, 1e-6)
    return npk == HKL0.shape[1]

# vmap over the second axis (j)
test_combo_right = jax.jit(jax.vmap(test_combo, in_axes=(None, None, 0, None)))  # BT varies

# vmap over the first axis (i)
test_combo_both = jax.jit(jax.vmap(test_combo_right,  in_axes=(0, 0, None, 0)))  # gobs and gt vary

@jax.jit
def derive_arrays(ha, hb, B, BI, HKL0):   
    BT = BTmat(ha, hb, B, BI)
    ga = jnp.dot(B, ha)
    gb = jnp.dot(B, hb)
    UBI = quickorient(ga, gb, BT)
    UB = jnp.linalg.inv(UBI)
    gtest = (UB @ HKL0).T
    return BT, ga, gb, gtest

derive_arrays_2d = jax.jit(jax.vmap(jax.vmap(derive_arrays, in_axes=[None, 0, None, None, None]), in_axes=[0, None, None, None, None]))

@jax.jit
def mask_hkl_indices(hkl_indices, disallow):
    i_idx, j_idx = hkl_indices
    n = i_idx.size  # flatten for iteration

    # flatten arrays for easier indexing
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()

    def body_fun(k, vals):
        i_arr, j_arr = vals
        i = i_arr[k]
        j = j_arr[k]
        # only update if i,j are not nan
        i_arr = i_arr.at[k].set(jnp.where(jnp.isnan(i) | jnp.isnan(j), i,
                                         jnp.where(disallow[i.astype(int), j.astype(int)], jnp.nan, i)))
        j_arr = j_arr.at[k].set(jnp.where(jnp.isnan(i) | jnp.isnan(j), j,
                                         jnp.where(disallow[i.astype(int), j.astype(int)], jnp.nan, j)))
        return i_arr, j_arr

    i_flat, j_flat = jax.lax.fori_loop(0, n, body_fun, (i_flat, j_flat))

    # reshape back
    return i_flat.reshape(i_idx.shape), j_flat.reshape(j_idx.shape)


@jax.jit
def filter_pairs(hkls1, hkls2, c2a, B, BI, tol=1e-5):
    """ remove duplicate pairs for orientation searches
    h1 = reflections of ring1, N1 peaks
    h2 = reflections of ring2, N2 peaks
    c2a  = cos angle between them, N1xN2
    B = B matrix in reciprocal space
    BI = inverse in real space
    """

    N1 = hkls1.shape[0]
    N2 = hkls2.shape[0]
    M = N1 * N2

    btmats, ga, gb, gtest = derive_arrays_2d(hkls1, hkls2, B, BI, HKL0)
    ga_flat = ga.reshape((M, 3))
    gb_flat = gb.reshape((M, 3))
    bt_flat = btmats.reshape((M, 3, 3))
    gt_flat = gtest.reshape((M, -1, 3))

    # test each (gobs and gtest) from the left with (bt) from the right
    adj = test_combo_both(ga_flat, gb_flat, bt_flat, gt_flat)
    # see if they're the same angle
    same_ang = jnp.isclose(c2a.ravel()[:, None], c2a.ravel()[None, :])

    both = adj & same_ang

    # get isolated components
    components = connected_components(both)
    components = jnp.where(components > -1, components, jnp.nan)
    hkl_indices = jnp.unravel_index(components, (hkls1.shape[0], hkls2.shape[0]))

    disallow = jnp.abs(c2a) > 0.98

    return jnp.array(mask_hkl_indices(hkl_indices, disallow)).T