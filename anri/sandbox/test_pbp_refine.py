import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "0"

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", 'cpu')

from ImageD11 import unitcell
from ImageD11.sinograms import geometry
from ImageD11.sinograms.point_by_point import PBPRefine, unitcell_to_b, PBPMap

import transform as mytrans
import index as myidx

os.chdir('../../tests/data/sandbox/refine/FeAu_0p5_tR_nscope/FeAu_0p5_tR_nscope_top_200um/')

refman = PBPRefine.from_h5('FeAu_0p5_tR_nscope_top_200um_refine_manager_Fe.h5')

os.chdir('../../../../../../anri/sandbox/')


@jax.jit
def step_to_recon(si, sj, recon_shape):
    """Converts step space (si, sj) to reconstruction space (ri, rj)"""
    ri = si + (recon_shape[0] // 2)
    rj = sj + (recon_shape[1] // 2)
    return ri, rj

@jax.jit
def get_voxel_idx(y0, xi0, yi0, omega, dty, ystep):
    """
    get peaks at xi0, yi0
    basically just geometry.dty_values_grain_in_beam_sincos
    """
    
    ydist = jnp.abs(y0 - xi0 * jnp.sin(omega) - yi0 * jnp.cos(omega) - dty)
    mask = ydist <= ystep

    return mask, ydist

@jax.jit
def weighted_lstsq_ubi_fit(ydist, gve, hkl, weight_mask):
    # run the weighted fit
    # a.T @ gve = h =>  gve.T @ a = h.T => a = np.linalg.pinv(gve.T) @ h.T, same for b and c
    w = (1. / (ydist + 1)).reshape(gve.shape[0], 1)
    w *= weight_mask.astype(gve.dtype)[:, None]
    a = w * gve
    b = w * hkl
    m, n = a.shape[-2:]
    rcond = jnp.finfo(b.dtype).eps * max(n, m)
    ubifitT, residuals, rank, sing_vals = jnp.linalg.lstsq(a, b, rcond=rcond)
    ubifit = ubifitT.T

    return w, ubifit, residuals, rank, sing_vals

@jax.jit
def divide_where(arr1, arr2, out, wherearr):
    """
    Do arr1/arr2.
    In locations where wherearr == 0, return out instead
    """
    div = jnp.divide(arr1, arr2)
    return jnp.where(wherearr != 0, div, out)

@jax.jit
def strain_fit(ydist, gve, hkl, U, B0, gnoise_std, weight_mask, eps_reg=1e-12):
    """Optimized strain fit.
    
    Shapes (assumed fixed across calls):
    - gve: (N, 3)
    - hkl: (N, 3)
    - ydist: (N,) or (N,1)
    - weight_mask: (N,) or (N,1)
    
    
    Returns:
    3x3 symmetric strain tensor (dtype matches inputs)
    """
    
    
    # Helper: robust division for small denominators
    def safe_div(num, den, default=0.0, eps=1e-12):
        den_pos = jnp.where(jnp.abs(den) > eps, den, jnp.nan)
        return jnp.where(jnp.isfinite(den_pos), num / den_pos, default)
    
    def remove_outliers(vec):
        mean = jnp.mean(vec)
        std = jnp.std(vec)
        lo = mean - 3.5 * std
        hi = mean + 3.5 * std
        return jnp.where((vec >= lo) & (vec <= hi), vec, 0.0)
    
    # gve0: projection of hkl through U.dot(B0)
    UB = jnp.dot(U, B0)
    # einsum is fine for this contraction; keeps shapes clear
    gve0 = jnp.einsum('ij,nj->ni', UB, hkl)
    
    gTg0 = jnp.sum(gve * gve0, axis=1)
    gTg = jnp.sum(gve * gve, axis=1)
    
    directional_strain = gTg0/gTg - 1
    
    # unit direction kappa (safe norm)
    gve_norm = jnp.linalg.norm(gve, axis=1, keepdims=True)
    # gve_norm = jnp.maximum(gve_norm, 1e-12)
    kappa = gve / gve_norm
    kx = kappa[:, 0]
    ky = kappa[:, 1]
    kz = kappa[:, 2]
    
    # Build design matrix M explicitly (N x 6)
    M = jnp.stack([
        kx * kx,
        ky * ky,
        kz * kz,
        2.0 * kx * ky,
        2.0 * kx * kz,
        2.0 * ky * kz,
    ], axis=1)
    
    # Base weights from geometry
    w = 1.0 / (ydist.reshape(-1) + 1.0)
    
    # Propagated noise in directional strain
    a_noise = jnp.sum(gve0 * (gnoise_std ** 2) * gve0, axis=1)
    strain_noise_std = jnp.sqrt(safe_div(a_noise, (gTg ** 2), default=1.0))
    
    # Combine weights and apply mask
    w = w / jnp.maximum(strain_noise_std, 1e-12)
    w = jnp.where(jnp.isfinite(w), w, 0.0)
    
    # Remove outliers more than 3.5 std away
    w = remove_outliers(w)

    # Apply provided mask (ensure same shape)
    wm = weight_mask.reshape(-1)
    w = w * wm.astype(gve.dtype)

    # normalize
    wmax = jnp.maximum(jnp.max(w), 1e-12)
    w = w / wmax
    
    # Weighted design and RHS
    A = (w[:, None] * M) # shape (N, 6)
    b = w * directional_strain # shape (N,) 
    
    # Solve normal equations with tiny Tikhonov regularization for stability
    ATA = A.T @ A # shape (6,6)
    ATb = A.T @ b # shape (6,)

    # scale-robust regularization: scale by trace of ATA
    trace = jnp.trace(ATA)
    scale = jnp.maximum(trace / 6.0, 1.0)
    reg = eps_reg * scale
    ATA_reg = ATA + reg * jnp.eye(6, dtype=ATA.dtype)
    
    eps_vec = jnp.linalg.solve(ATA_reg, ATb) # shape (6,)
    
    # Map to symmetric strain tensor
    sxx, syy, szz, sxy, sxz, syz = eps_vec
    eps_tensor = jnp.array([
        [sxx, sxy, sxz],
        [sxy, syy, syz],
        [sxz, syz, szz],
    ], dtype=eps_vec.dtype)
    
    # scale-robust regularization: scale by trace of ATA
    return eps_tensor


@jax.jit
def ubi_to_unitcell(ubi):
    mt = jnp.dot(ubi, ubi.T)
    G = mt
    a, b, c = jnp.sqrt(jnp.diag(G))
    al = jnp.degrees(jnp.arccos(G[1, 2] / b / c))
    be = jnp.degrees(jnp.arccos(G[0, 2] / a / c))
    ga = jnp.degrees(jnp.arccos(G[0, 1] / a / b))
    return jnp.array([a, b, c, al, be, ga])

@jax.jit
def ubi_and_ucell_to_u(ubi, ucell):
    # compute B
    a, b, c = ucell[:3]
    ralpha, rbeta, rgamma = jnp.radians(ucell[3:])  # radians
    ca = jnp.cos(ralpha)
    cb = jnp.cos(rbeta)
    cg = jnp.cos(rgamma)
    g = jnp.full((3, 3), jnp.nan, jnp.float64)
    g.at[0, 0].set(a * a)
    g.at[0, 1].set(a * b * cg)
    g.at[0, 2].set(a * c * cb)
    g.at[1, 0].set(a * b * cg)
    g.at[1, 1].set(b * b)
    g.at[1, 2].set(b * c * ca)
    g.at[2, 0].set(a * c * cb)
    g.at[2, 1].set(b * c * ca)
    g.at[2, 2].set(c * c)
    gi = jnp.linalg.inv(g)
    astar, bstar, cstar = jnp.sqrt(jnp.diag(gi))
    betas = jnp.degrees(jnp.arccos(gi[0, 2] / astar / cstar))
    gammas = jnp.degrees(jnp.arccos(gi[0, 1] / astar / bstar))

    B = jnp.zeros((3, 3))

    B.at[0, 0].set(astar)
    B.at[0, 1].set(bstar * jnp.cos(jnp.radians(gammas)))
    B.at[0, 2].set(cstar * jnp.cos(jnp.radians(betas)))
    B.at[1, 0].set(0.0)
    B.at[1, 1].set(bstar * jnp.sin(jnp.radians(gammas)))
    B.at[1, 2].set(-cstar * jnp.sin(jnp.radians(betas)) * ca)
    B.at[2, 0].set(0.0)
    B.at[2, 1].set(0.0)
    B.at[2, 2].set(1.0 / c)

    u = jnp.dot(B, ubi).T
    return u


# Suppose gve_all is a huge numpy array (50+ GB)
# Wrap refine_pixel so gve_all is captured in a closure
def make_refine_pixel(gve_all):
    def refine_pixel(ubi_in, ri, rj):
        refine_here = jnp.logical_and(recon_mask[ri, rj],
                                      ~jnp.isnan(ubi_in[0,0]))

        def do_refine(u):
            xi0 = sx_grid[ri, rj]
            yi0 = sy_grid[ri, rj]

            sino_mask, ydist = get_voxel_idx(y0, xi0, yi0, omega, dty, ystep)

            hkl, hkl_int = myidx.gv_ubi_to_hkl_int(u, gve_all)
            dhkl = hkl - hkl_int
            drlv2 = jnp.sum(dhkl**2, axis=1)

            score_mask = (drlv2 < hkl_tol**2) & sino_mask

            w, ubifit, residuals, rank, sing_vals = weighted_lstsq_ubi_fit(ydist, gve_all, hkl_int, score_mask)

            score_mask_post_refine = myidx.score_mask(ubifit, gve_all, hkl_tol) & sino_mask
            score_post_refine = jnp.sum(score_mask_post_refine)

            ucell = ubi_to_unitcell(ubifit)
            U = ubi_and_ucell_to_u(ubifit, ucell)

            eps_tensor = strain_fit(ydist, gve_all, hkl_int, U, B0, gnoise_std, score_mask_post_refine)

            return ubifit, eps_tensor, score_post_refine

        def skip_refine(u):
            return (jnp.full((3,3), jnp.nan), jnp.full((3,3), jnp.nan), 1)

        return jax.lax.cond(refine_here, do_refine, skip_refine, operand=ubi_in)

    return jax.jit(refine_pixel)  # JIT compile with gve_all captured


self = refman

# columnfile by [3, 3, (ri, rj)]
all_pbpmap_ubis = self.pbpmap.ubi

pars = self.icolf.parameters.get_parameters()

dummy_var = np.eye(3)
uc = unitcell.unitcell_from_parameters(self.icolf.parameters)
B0 = unitcell_to_b(uc.lattice_parameters, dummy_var)

origin_sample = np.zeros((self.icolf.nrows, 3))
origin_sample[:, 0] = self.icolf.xpos_refined

xyz_lab = np.column_stack((self.icolf.xl,
                  self.icolf.yl,
                  self.icolf.zl))

gve_all = jax.device_get(mytrans.xyz_lab_to_g(xyz_lab, self.icolf.omega, origin_sample, pars.get('wedge'), pars.get('chi'), pars.get('wavelength')))

dty = self.icolf.dty
ystep = self.ystep
y0 = self.y0
hkl_tol = self.hkl_tol_refine
si = self.pbpmap.i
sj = self.pbpmap.j
recon_mask = jnp.array(self.mask)
recon_shape = recon_mask.shape
ri, rj = step_to_recon(si, sj, recon_shape)
ri = np.round(ri).astype(int)
rj = np.round(rj).astype(int)
sx_grid = jnp.array(self.sx_grid)
sy_grid = jnp.array(self.sy_grid)
omega = self.icolf.omega
gnoise_std = 1e-4

ubit = jnp.transpose(all_pbpmap_ubis, (2, 0, 1))

refine_pixel_fn = make_refine_pixel(gve_all)
refine_pixel_batch = jax.vmap(refine_pixel_fn, in_axes=(0,0,0))
print('function call')
ubis_out, eps_out, scores_out = refine_pixel_batch(ubit, ri, rj)
print('trying to print result')
print(ubis_out[-1])