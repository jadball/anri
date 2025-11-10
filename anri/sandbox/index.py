import jax
import jax.numpy as jnp
from jax import jit, vmap

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")


# very basic utilites
@jit
def unit(v):
    """Normalize a single vector."""
    return v / jnp.linalg.norm(v)


unit_many = jit(vmap(unit))


@jit
def _g_to_ds(gvec):
    """Compute d-star from g-vector (just the length).

    Args:
        gvec: (3,) g-vector

    Returns:
        ds: scalar d-star
    """
    ds = jnp.linalg.norm(gvec)
    return ds


g_to_ds = jit(vmap(_g_to_ds))


@jit
def assign_peaks_to_rings(ds, ringds, ds_tol):
    """Assign g-vectors (by d-star) to rings based on closest ring d-star within tolerance.

    Args:
        ds: (Npks,) array of d-star values for g-vectors
        ringds: (Nrings,) array of theoretical ring d-star values
        ds_tol: scalar tolerance for assignment

    Returns:
        Array of shape (Npks,) with index of assigned ring for each g-vector, or -1 if no assignment.
    """
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
    """Compute cos(angles) between all pairs of reciprocal lattice vectors in hkl1 and hkl2.

    JAX JIT implementation of ImageD11.unitcell.anglehkls()

    Args:
        hkl1: (N,3) array of hkls in ring 1
        hkl2: (M,3) array of hkls in ring 2
        rmt: (3,3) reciprocal metric tensor
    Returns:
        cosines: (N,M) array of cosines
    """
    # Compute squared norms under metric
    g1 = jnp.einsum("ni,ij,nj->n", hkl1, rmt, hkl1)  # (N,)
    g2 = jnp.einsum("mi,ij,mj->m", hkl2, rmt, hkl2)  # (M,)

    # Compute dot products under metric
    g12 = jnp.einsum("ni,ij,mj->nm", hkl1, rmt, hkl2)  # (N,M)

    # Compute cosines
    cos_theta = g12 / jnp.sqrt(g1[:, None] * g2[None, :])

    return cos_theta


@jax.jit
def get_rotmat(v1, v2):
    """Build orthonormal frame from v1, v2.

    JAX JIT implementation of ImageD11.cImageD11.quickorient(UBI, BT)

    Args:
        v1: (3,) first vector
        v2: (3,) second vector

    Returns:
        [u1, u2, u3] (3x3)
    """
    g3 = jnp.cross(v1, v2)
    u1 = unit(v1)
    u3 = unit(g3)
    u2 = jnp.cross(u1, u3)
    M = jnp.array((u1, u2, u3))
    return M


@jax.jit
def BTmat(h1, h2, B, BI):
    """Generate the B-transpose matrix from two hkl vectors and B, BI matrices.

    JAX JIT implementation of ImageD11.unitcell.BTmat()

    Reference:
        W. R. Busing and H. A. Levy. 457. Acta Cryst. (1967). 22, 457-464
        https://doi.org/10.1107/S0365110X67000970

    Args:
        h1: (3,) first hkl vector
        h2: (3,) second hkl vector
        B: B matrix (3x3)
        BI: Inverse B matrix (3x3)

    Returns:
        B-transpose matrix (3x3)
    """
    g1 = B @ h1
    g2 = B @ h2
    M = get_rotmat(g1, g2)
    return BI @ M.T


@jax.jit
def quickorient(g1, g2, BT):
    """Get UBI matrix from two g-vectors and B-transpose matrix.

    JAX JIT implementation of ImageD11.cImageD11.quickorient(UBI, BT).
    See get_rotmat() for details.

    After:
    W. R. Busing and H. A. Levy. 457. Acta Cryst. (1967). 22, 457-464
    https://doi.org/10.1107/S0365110X67000970

    Args:
        g1: (3,) first g-vector
        g2: (3,) second g-vector
        BT: B-transpose matrix (3x3)

    Returns:
        UBI matrix (3x3)
    """
    M = get_rotmat(g1, g2)
    return BT @ M


@jax.jit
def gv_ubi_to_hkl(ubi, gv):
    # Transform gv by ubi
    # ubi: (3,3)
    # gv: (N,3)
    gv = jnp.atleast_2d(gv)  # (N,3)
    hkl = jnp.einsum('ij,nj->ni', ubi, gv)

    return hkl

@jax.jit
def gv_ubi_to_hkl_int(ubi, gv):
    hkl = gv_ubi_to_hkl(ubi, gv)
    hkl_int = jnp.round(hkl)
    return hkl, hkl_int

@jax.jit
def score_mask(ubi, gv, tol):
    """Make a mask of g-vectors indexed by UBI within tolerance.

    JAX JIT implementation of ImageD11.cImageD11.score()

    Args:
        ubi: (3,3) orientation matrix
        gv: (N,3) array of g-vectors
        tol: scalar tolerance

    Returns:
        mask: (N,) mask where g-vectors index the UBI
    """
    hkl, hkl_int = gv_ubi_to_hkl_int(ubi, gv)
    dhkl = hkl - hkl_int

    # Squared distance
    drlv2 = jnp.sum(dhkl**2, axis=1)

    mask = drlv2 < tol**2
    return mask


@jax.jit
def score(ubi, gv, tol):
    # get mask where gv scored by ubi
    sm = score_mask(ubi, gv, tol)

    # Count peaks where drlv2 < tol^2
    n = jnp.sum(sm)
    return n


@jax.jit
def find_gvector_cosangle_matches(
    gv: jnp.ndarray,  # (N,3)
    ra: jnp.ndarray,  # (N,)
    ring1: int,
    ring2: int,
    hkls1: jnp.ndarray,  # (M,3)
    hkls2: jnp.ndarray,  # (K,3)
    rmt,
    tol: float = 1e-5,
):
    """Find pairs of g-vectors from two rings that match angles between hkls within a tolerance.

    JAX JIT implementation of ImageD11.indexing.find()

    Args:
        gv: (N, 3) array of g-vectors
        ra: (N,) array of ring assignments for each g-vector
        ring1: ID of first ring
        ring2: ID of second ring
        hkls1: (M, 3) array of theoretical hkl reflections for ring1
        hkls2: (K, 3) array of theoretical hkl reflections for ring2
        rmt: (3, 3) reciprocal metric tensor
        tol: Maximum allowed difference in cos(angle). Defaults to 1e-5.

    Returns:
        Pytree of arrays:
            diffs: (N,) array of best cos(angle) differences for each g-vector in ring1
            i1: (N,) array of indices of g-vectors in ring1
            i2: (N,) array of indices of best-matching g-vectors in ring2 - doesn't consider tolerance
            mask: (N,) boolean array indicating valid matches within tolerance
    """
    N = gv.shape[0]

    cosines_theoretical = anglehkls(hkls1, hkls2, rmt).reshape(-1)  # shape M*K

    # Mark cosines that are too close to ±1 as NaN to avoid numerical issues
    bad = (jnp.abs(cosines_theoretical - 1.0) < 1e-5) | (jnp.abs(cosines_theoretical + 1.0) < 1e-5)
    coses = jnp.where(bad, jnp.nan, cosines_theoretical)  # shape stays M*K

    # Normalize gv
    gv_n = unit_many(gv)

    # Masks for ring membership
    mask1 = ra == ring1  # (N,)
    mask2 = ra == ring2  # (N,)

    # Mask out non-ring entries with NaN
    n1 = jnp.where(mask1[:, None], gv_n, jnp.nan)  # (N,3)
    n2 = jnp.where(mask2[:, None], gv_n, jnp.nan)  # (N,3)

    # Pairwise dot products (N,N) with NaNs for other rings
    costheta = n1 @ n2.T  # (N,N)

    # |costheta - theoretical|  → (N,N,M*K)
    diff_matrix = jnp.abs(costheta[:, :, None] - coses[None, None, :])

    # Best match difference for each (i,j)
    best_diff_ij = jnp.nanmin(diff_matrix, axis=2)  # (N,N)

    # Best j per i - does not consider tolerance
    best_j = jnp.nanargmin(best_diff_ij, axis=1)  # (N,)
    best_diff = jnp.nanmin(best_diff_ij, axis=1)  # (N,)

    # Which diffs are within tolerance?
    hit_mask = mask1 & (best_diff < tol)  # (N,)

    return {
        "diffs": best_diff.astype(jnp.float32),  # (N,)
        "i1": jnp.arange(N, dtype=jnp.int32),  # (N,)
        "i2": best_j.astype(jnp.int32),  # (N,)
        "mask": hit_mask.astype(jnp.bool_),  # (N,)
    }


@jax.jit
def connected_components(adj):
    """Computes the connected components of an undirected graph represented by an adjacency matrix.

    This function uses the Union-Find algorithm with path compression to determine the connected
    components of the graph. The graph is represented as an adjacency matrix, where `adj[i, j] == 1`
    indicates an edge between nodes `i` and `j`.

    How it works:
    1. Each node is initially its own parent.
    2. The `find` function is used to locate the root parent of a node, applying path compression
    to optimize future lookups.
    3. The `union` function merges two connected components by setting the parent of one component's
    root to the other.
    4. The outer loop iterates over all nodes, and for each node, an inner loop checks its neighbors
    in the adjacency matrix. If two nodes are connected, their components are merged.
    5. After processing all nodes, the `compress_parent` step ensures that all nodes point directly
    to their root parent for consistency.
    6. Finally, the unique parent nodes are returned, representing the connected components.

    Args:
        adj (jax.numpy.ndarray): A square adjacency matrix of shape (n, n), where `n` is the number
            of nodes in the graph. The matrix should contain binary values (0 or 1).

    Returns:
        jax.numpy.ndarray: A 1D array of size `n` containing the representative parent node for each
            connected component. Nodes in the same connected component will have the same parent.
            If the number of unique components is less than `n`, the remaining entries are filled
            with `-1`.
    """
    n = adj.shape[0]
    parents = jnp.arange(n)

    def find(parents, x):
        # Path compression
        def loop_body(val):
            px = val
            return parents[px]

        def cond(val):
            return parents[val] != val

        return jax.lax.while_loop(cond, loop_body, x)

    def union(parents, x, y):
        px = find(parents, x)
        py = find(parents, y)
        parents = parents.at[px].set(py)
        return parents

    def body_fun(i, parents):
        def inner_body(j, parents):
            parents = jax.lax.cond(adj[i, j] == 1, lambda p: union(p, i, j), lambda p: p, parents)
            return parents

        parents = jax.lax.fori_loop(0, n, inner_body, parents)
        return parents

    parents = jax.lax.fori_loop(0, n, body_fun, parents)

    # Flatten parents
    def compress_parent(i, parents):
        return parents.at[i].set(find(parents, i))

    parents = jax.lax.fori_loop(0, n, compress_parent, parents)

    return jnp.unique(parents, size=n, fill_value=-1)


@jax.jit
def filter_pairs(hkls1, hkls2, c2a, B, BI):
    """Remove duplicate pairs for orientation searches.

    To define a UBI, you need two g-vectors and the corresponding BT matrix from a pair of HKLs.
    We know the theoretical hkls for each ring, but we don't know which hkls exactly the g-vectors are.
    The first test we can do is whether the angle between the g-vectors matches the angle between the HKLs.
    However, this is not sufficient - two pairs of hkls can generate the same cos(angle), but define different unique UBIs.
    To resolve this, we check each hkl pair against each other hkl pair.
    Two hkl pairs are considered equivalent if they have the same cosine angle, and if the UBI from one pair index the g-vectors from the other pair.
    This function finds all unique pairs of hkls from hkls1 and hkls2 that pass these tests.

    JAX JIT implementation of ImageD11.unitcell.filter_pairs()

    Args:
        hkls1: (N1, 3) array of hkls for ring 1
        hkls2: (N2, 3) array of hkls for ring 2
        c2a: (N1, N2) array of cos(angle) between hkls1 and hkls2
        B: (3,3) B matrix
        BI: (3,3) Inverse B matrix

    Returns:
        hab: (M, 6) array of unique hkl pairs (ha, hb) from hkls1 and hkls2 with duplicate pairs masked to jnp.nan
        unique_angles: (M,) array of cos(angle) for each of the unique pairs with duplicate pairs masked to jnp.nan
        unique_matrices: (M, 3, 3) array of B-transpose matrices for each of the unique pairs with duplicate pairs masked to jnp.nan
    """
    HKL0 = jnp.array(
        [
            [0, 0, 1, 1, -1, 1, -1, 0, 0, 1, -1, 1, 1, 3, 11],
            [0, 1, 0, 1, 1, 0, 0, 1, -1, 1, 1, -1, 1, 2, 12],
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 1, 13],
        ]
    )  # test hkls for determining UBI uniqueness

    @jax.jit
    def derive_arrays(ha, hb, B, BI):  # , HKL0):
        """Compute BT, ga, gb, and gtest arrays for a given hkl pair.

        We go from hkl to g-vectors (ga, gb) using B matrix.
        We compute BT from the hkl pair and B, BI.
        We compute UBI from (ga, gb) and BT.
        We invert UBI to get UB.
        We transform HKL0 by UB to get gtest.

        Args:
            ha: (3,) First hkl vector
            hb: (3,) Second hkl vector
            B: (3,3) B matrix
            BI: (3,3) Inverse B matrix

        Returns:
            BT: (3,3) B-transpose matrix
            ga: (3,) g-vector for ha
            gb: (3,) g-vector for hb
            gtest: (Ntest, 3) g-vectors from HKL0 transformed
        """
        BT = BTmat(ha, hb, B, BI)
        ga = jnp.dot(B, ha)
        gb = jnp.dot(B, hb)
        UBI = quickorient(ga, gb, BT)
        UB = jnp.linalg.inv(UBI)
        gtest = (UB @ HKL0).T
        return BT, ga, gb, gtest

    @jax.jit
    def test_combo(ga, gb, BT, gt):
        """Test if UBI from hkl pairs is equivalent.

        Tests if UBI from (ga, gb, BT) (one pair) indexes all g-vectors in gt (from the other pair).

        Args:
            ga: (3,) First g-vector from the first pair
            gb: (3,) Second g-vector from the first pair
            BT: (3,3) B-transpose matrix from the second pair
            gt: (Ntest, 3) g-vectors from HKL0 transformed by UB from the second pair

        Returns:
            True if all g-vectors in gt are indexed by UBI from (ga, gb, BT), false otherwise
        """
        UBI = quickorient(ga, gb, BT)
        npk = score(UBI, gt, 1e-6)
        return npk == HKL0.shape[1]

    # vmap over the second axis (j)
    test_combo_right = jax.jit(jax.vmap(test_combo, in_axes=(None, None, 0, None)))  # BT varies

    # vmap over the first axis (i)
    test_combo_both = jax.jit(jax.vmap(test_combo_right, in_axes=(0, 0, None, 0)))  # gobs and (ga, gb) vary

    derive_arrays_2d = jax.jit(
        jax.vmap(jax.vmap(derive_arrays, in_axes=[None, 0, None, None]), in_axes=[0, None, None, None])
    )

    N1 = hkls1.shape[0]
    N2 = hkls2.shape[0]
    M = N1 * N2

    # Derive required arrays for each hkl pair
    btmats, ga, gb, gtest = derive_arrays_2d(hkls1, hkls2, B, BI)  # (N1, N2, ...)
    # Flatten results so we can test pairs against other pairs
    ga_flat = ga.reshape((M, 3))
    gb_flat = gb.reshape((M, 3))
    bt_flat = btmats.reshape((M, 3, 3))
    gt_flat = gtest.reshape((M, -1, 3))

    # test each (gobs and gtest) from the left with (bt) from the right
    # True if the UBIs are equivalent
    adj = test_combo_both(ga_flat, gb_flat, bt_flat, gt_flat)
    # See if cos(angle) are also the same
    same_ang = jnp.isclose(c2a.ravel()[:, None], c2a.ravel()[None, :])

    both = adj & same_ang

    # Get connected components of the equivalence graph - -1 indicates invalid components
    components = connected_components(both)

    # Build mask for valid components
    valid_mask = components >= 0

    # Replace invalid with 0 to keep indices in-bounds
    safe_components = jnp.where(valid_mask, components, 0)

    # Unravel all components (same shape as components)
    i_idx, j_idx = jnp.unravel_index(safe_components, (hkls1.shape[0], hkls2.shape[0]))

    # Gather corresponding values
    angles_all = c2a[i_idx, j_idx]
    mats_all = btmats[i_idx, j_idx]

    ha = hkls1[i_idx]
    hb = hkls2[j_idx]

    # Mask duplicate components to nan
    unique_angles = jnp.where(valid_mask, angles_all, jnp.nan)
    unique_matrices = jnp.where(valid_mask[:, None, None], mats_all, jnp.nan)
    ha = jnp.where(valid_mask[:, None], ha, jnp.nan)
    hb = jnp.where(valid_mask[:, None], hb, jnp.nan)

    # now mask the unique angles to disallow cosangle close to 1
    mask_angle_too_close = jnp.abs(unique_angles) > 0.98
    # mask unique_angles and unique_matrices
    unique_angles = jnp.where(~mask_angle_too_close, unique_angles, jnp.nan)
    unique_matrices = jnp.where(~mask_angle_too_close[:, None, None], unique_matrices, jnp.nan)
    ha = jnp.where(~mask_angle_too_close[:, None], ha, jnp.nan)
    hb = jnp.where(~mask_angle_too_close[:, None], hb, jnp.nan)

    # sort by increasing angle:
    sort_indices = jnp.argsort(jnp.nan_to_num(unique_angles, nan=jnp.inf))
    unique_angles = unique_angles[sort_indices]
    unique_matrices = unique_matrices[sort_indices]
    ha = ha[sort_indices]
    hb = hb[sort_indices]

    hab = jnp.concatenate([ha, hb], axis=1)

    return hab, unique_angles, unique_matrices
