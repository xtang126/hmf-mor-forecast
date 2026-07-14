"""
hmf_mor_forecast.py
===================

Forecast likelihood for cluster number counts with a mass-observable relation
(MOR) with log-normal scatter.

Model
-----
1. Halo mass function n(M, z) from `hmf` (Tinker08 by default).
2. MOR:  ln O = ln A + alpha*ln(M/M_piv) + beta*ln((1+z)/(1+z_piv)) + eps,
         eps ~ N(0, sigma_lnO^2).
3. Inverted: at fixed true M, the inferred mass has Gaussian scatter in log10:
         sigma_logM = sigma_lnO / (alpha * ln 10),
         with sigma_lnO ~ f_obs for small fractional observable scatter.
4. Migration matrix  P_ij = P(observed bin i | true bin j), columns normalised.
5. Poisson likelihood between observed data counts and scattered model counts.

Two migration matrices are stored:
  - P_data:  the TRUTH used to generate the mock (frozen in setup)
  - P_model: what the analyst assumes. Can equal P_data (correct assumption),
             be the identity (ignore scatter), or be rebuilt each MCMC step
             from a sampled nuisance parameter (marginalise over scatter).
"""

import numpy as np
from functools import lru_cache
from math import erf, sqrt

from cosmosis.datablock import names, option_section
from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H_LITTLE = 0.7
H0 = H_LITTLE * 100.0        # km/s/Mpc
FULL_SKY_DEG2 = 41253.0


# ---------------------------------------------------------------------------
# Comoving volume of a redshift shell (returned in (Mpc/h)^3)
# ---------------------------------------------------------------------------
def _volume_shell(zmin, zmax, Om0, area_deg2):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    V = (cosmo.comoving_volume(zmax) - cosmo.comoving_volume(zmin)).to(u.Mpc**3).value
    V /= H_LITTLE**3
    return V * (area_deg2 / FULL_SKY_DEG2)


@lru_cache(maxsize=2000)
def _volume_shell_cached(z1, z2, om_r, area_r):
    return _volume_shell(float(z1), float(z2), float(om_r), float(area_r))


def volume_shell(zmin, zmax, Om0, area_deg2):
    return _volume_shell_cached(
        round(zmin, 4), round(zmax, 4),
        round(Om0, 5),  round(area_deg2, 3),
    )


# ---------------------------------------------------------------------------
# MOR: convert fractional observable scatter to sigma_log10(M)
# ---------------------------------------------------------------------------
def sigma_logM_from_MOR(frac_obs_scatter, alpha_MOR):
    """
    sigma_log10(M) = sigma_lnO / (alpha * ln 10).
    For small fractional scatter, sigma_lnO ~ frac_obs_scatter.
    """
    if alpha_MOR <= 0:
        raise ValueError("alpha_MOR must be positive")
    return float(frac_obs_scatter) / (alpha_MOR * np.log(10.0))


# ---------------------------------------------------------------------------
# Migration matrix
# ---------------------------------------------------------------------------
def _gauss_cdf(x, mu, sigma):
    return 0.5 * (1.0 + erf((x - mu) / (sqrt(2.0) * sigma)))


def build_migration_matrix(M_edges_logh, sigma_logM_per_bin):
    """
    P[i, j] = Prob(true-bin-j halo is observed in bin i). Columns normalised.
    sigma_logM_per_bin: scalar or array of length n_bins (dex).
    Zero/negative scatter -> identity (no migration).
    """
    edges = np.asarray(M_edges_logh, dtype=float)
    centres = 0.5 * (edges[1:] + edges[:-1])
    nb = centres.size

    sig = np.atleast_1d(np.asarray(sigma_logM_per_bin, dtype=float))
    if sig.size == 1:
        sig = np.full(nb, sig.item())
    if sig.size != nb:
        raise ValueError("sigma_logM_per_bin length must equal number of bins")

    if np.all(sig <= 0):
        return np.eye(nb)

    P = np.zeros((nb, nb))
    for j in range(nb):
        mu, s = centres[j], sig[j]
        if s <= 0:
            P[j, j] = 1.0
            continue
        for i in range(nb):
            P[i, j] = _gauss_cdf(edges[i + 1], mu, s) - _gauss_cdf(edges[i], mu, s)
        col = P[:, j].sum()
        if col > 0:
            P[:, j] /= col
    return P


# ---------------------------------------------------------------------------
# True counts per mass bin at given cosmology
# ---------------------------------------------------------------------------
def compute_true_counts(mf, M_edges_logh, V, z_mid, omegam=None, sigma8=None):
    edges = np.asarray(M_edges_logh)

    if (omegam is not None) or (sigma8 is not None):
        mf.update(
            z=z_mid,
            sigma_8=sigma8,
            cosmo_params={"H0": H0, "Om0": omegam},
            Mmin=edges[0],
            Mmax=edges[-1],
        )

    N = np.zeros(len(edges) - 1)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mf.update(Mmin=lo, Mmax=hi)
        if len(mf.m) < 2:
            continue
        dM = np.gradient(mf.m)
        N[k] = np.sum(mf.dndm * dM) * V   # (h^3 Mpc^-3) * (h^-3 Mpc^3)
    return N


# ---------------------------------------------------------------------------
# cosmosis setup
# ---------------------------------------------------------------------------
def setup(options):
    # --- Fiducial cosmology (mock generation only) ---
    Om0_fid    = options.get_double(option_section, "Om0_fid",    default=0.318)
    sigma8_fid = options.get_double(option_section, "sigma8_fid", default=0.80)

    # --- Redshift bin ---
    zmin = options.get_double(option_section, "z_min", default=0.3)
    zmax = options.get_double(option_section, "z_max", default=0.5)
    z_mid = 0.5 * (zmin + zmax)

    # --- Survey area ---
    area_deg2 = options.get_double(option_section, "area_deg2", default=4000.0)

    # --- Mass range and binning (log10 Msun/h) ---
    mmin = options.get_double(option_section, "mass_min", default=2e14)
    mmax = options.get_double(option_section, "mass_max", default=5e15)
    if not (mmax > mmin > 0):
        raise ValueError("mass_min/mass_max must be positive and mass_max > mass_min")
    n_mass_bins = options.get_int(option_section, "n_mass_bins", default=10)
    M_edges_logh = np.linspace(np.log10(mmin), np.log10(mmax), n_mass_bins + 1)

    # --- MOR: fractional observable scatter and mass slope ---
    alpha_MOR   = options.get_double(option_section, "alpha_MOR",           default=1.0)
    frac_data   = options.get_double(option_section, "frac_scatter_data",   default=0.10)
    frac_model  = options.get_double(option_section, "frac_scatter_model",  default=0.10)
    marginalise = options.get_bool  (option_section, "marginalise_scatter", default=False)

    sigma_logM_data  = sigma_logM_from_MOR(frac_data,  alpha_MOR)
    sigma_logM_model = sigma_logM_from_MOR(frac_model, alpha_MOR)

    # --- MassFunction: one instance, updated per sample ---
    mf = MassFunction(
        z=z_mid,
        sigma_8=sigma8_fid,
        cosmo_params={"H0": H0, "Om0": Om0_fid},
        Mmin=M_edges_logh[0],
        Mmax=M_edges_logh[-1],
        dlog10m=0.1,
        hmf_model="Tinker08",
    )

    # --- Mock data (TRUE scatter) ---
    V_fid = volume_shell(zmin, zmax, Om0_fid, area_deg2)
    N_true_fid = compute_true_counts(mf, M_edges_logh, V_fid, z_mid)
    P_data = build_migration_matrix(M_edges_logh, sigma_logM_data)
    N_obs = np.clip(P_data @ N_true_fid, np.finfo(float).eps, None)

    # --- Model migration matrix: fixed, or rebuilt in execute if marginalising ---
    P_model_fixed = build_migration_matrix(M_edges_logh, sigma_logM_model)

    # --- Sanity print ---
    print("[hmf_mor_forecast] ================ setup ================")
    print(f"  z in [{zmin:.3f}, {zmax:.3f}],  area = {area_deg2:.0f} deg^2")
    print(f"  mass in [{mmin:.2e}, {mmax:.2e}] Msun/h,  n_bins = {n_mass_bins}")
    print(f"  alpha_MOR          = {alpha_MOR:.3f}")
    print(f"  frac_scatter_data  = {frac_data:.3f}  -> sigma_logM = {sigma_logM_data:.4f} dex")
    print(f"  frac_scatter_model = {frac_model:.3f}  -> sigma_logM = {sigma_logM_model:.4f} dex")
    print(f"  marginalise_scatter = {marginalise}")
    print(f"  true      total counts (fid): {N_true_fid.sum():.1f}")
    print(f"  observed  total counts (fid): {N_obs.sum():.1f}")
    print("[hmf_mor_forecast] =======================================")

    return {
        "zmin": zmin,
        "zmax": zmax,
        "z_mid": z_mid,
        "area_deg2": area_deg2,
        "M_edges_logh": M_edges_logh,
        "mf": mf,
        "N_obs": N_obs,
        "P_data": P_data,
        "P_model_fixed": P_model_fixed,
        "alpha_MOR": alpha_MOR,
        "marginalise": marginalise,
    }


# ---------------------------------------------------------------------------
# cosmosis execute
# ---------------------------------------------------------------------------
def execute(block, config):
    omegam = block[names.cosmological_parameters, "omega_m"]
    sigma8 = block[names.cosmological_parameters, "sigma8_input"]

    # Model migration matrix: fixed, or rebuilt from a sampled nuisance param
    if config["marginalise"]:
        frac_model = block[names.cosmological_parameters, "frac_scatter"]
        sig_logM   = sigma_logM_from_MOR(frac_model, config["alpha_MOR"])
        P_model    = build_migration_matrix(config["M_edges_logh"], sig_logM)
    else:
        P_model = config["P_model_fixed"]

    # Volume at this cosmology
    V = volume_shell(config["zmin"], config["zmax"], omegam, config["area_deg2"])

    # True model counts and scattered model counts
    N_true = compute_true_counts(
        config["mf"], config["M_edges_logh"], V, config["z_mid"],
        omegam=omegam, sigma8=sigma8,
    )
    N_model = np.clip(P_model @ N_true, 1e-12, None)

    # Poisson log-likelihood
    N_obs = config["N_obs"]
    loglike = float(np.sum(N_obs * np.log(N_model) - N_model - gammaln(N_obs + 1.0)))

    block["likelihoods", "hmf_like"] = loglike
    return 0


def cleanup(config):
    return 0