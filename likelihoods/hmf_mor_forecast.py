# scatter_massde.py
# Example config file: scatter_massde.ini

'''
Custom Poisson likelihood for halo mass function:
- Compares cluster counts to HMF prediction (Tinker08, hmf package)
- Assumes selection function = 1
- INCLUDES mass–observable scatter in both data and model
- Uses cached MassFunction and precomputed quantities for speed

Contact: Xin Tang (xt52@sussex.ac.uk)
'''

import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss

from cosmosis.datablock import names, option_section
from hmf import MassFunction
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import quad
from scipy.special import gammaln
from math import erf, sqrt

h = 0.7
H0 = h * 100  # km/s/Mpc


def volume_shell(zmin, zmax, Om0, area_deg2):
    """Return h^-3 Mpc^3 so it pairs with h^3 Mpc^-3 number densities."""
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    V = (cosmo.comoving_volume(zmax) - cosmo.comoving_volume(zmin)).to(u.Mpc**3).value
    V /= h**3                            # -> h^-3 Mpc^3
    f_sky = area_deg2 / 41253.0
    return V * f_sky


# ---- cached comoving volume shell ----
def V_shell_cached(zmin, zmax, Om0, area_deg2):
    return _V_shell_cached(
        round(zmin, 4), round(zmax, 4),
        round(Om0, 5),  round(area_deg2, 3)
    )


@lru_cache(maxsize=2000)
def _V_shell_cached(z1, z2, om_r, area_r):
    cosmo = FlatLambdaCDM(H0=H0, Om0=float(om_r))
    V = (cosmo.comoving_volume(float(z2)) - cosmo.comoving_volume(float(z1))).to(u.Mpc**3).value
    return V / h**3 * (float(area_r) / 41253.0)


def integrate_fixed(n_z, zmin, zmax, N=4):
    x, w = leggauss(N)
    z_nodes = 0.5 * (zmax - zmin) * x + 0.5 * (zmax + zmin)
    weights = 0.5 * (zmax - zmin) * w
    return np.sum(weights * np.array([n_z(z) for z in z_nodes]))


def gaussian_cdf(x, mu, sigma):
    # scalar Gaussian CDF in log10(M)
    t = (x - mu) / (sqrt(2.0) * sigma)
    return 0.5 * (1.0 + erf(t))


def build_migration_matrix(M_edges_logh, sigma_logM_per_bin):
    """
    P[i, j] = Prob(halo in true bin j is *observed* in bin i).
    Columns j normalised to 1.

    sigma_logM_per_bin[j] = scatter (dex) for true bin j.
    """
    M_edges_logh = np.asarray(M_edges_logh)
    bin_centres = 0.5 * (M_edges_logh[1:] + M_edges_logh[:-1])
    n_bins = len(bin_centres)

    sigma_logM_per_bin = np.asarray(sigma_logM_per_bin, dtype=float)
    if sigma_logM_per_bin.shape[0] != n_bins:
        raise ValueError("sigma_logM_per_bin must have length n_bins")

    P = np.zeros((n_bins, n_bins))

    for j in range(n_bins):
        mu = bin_centres[j]
        sigma_logM = sigma_logM_per_bin[j]

        for i in range(n_bins):
            lo = M_edges_logh[i]
            hi = M_edges_logh[i + 1]
            cdf_hi = gaussian_cdf(hi, mu, sigma_logM)
            cdf_lo = gaussian_cdf(lo, mu, sigma_logM)
            P[i, j] = cdf_hi - cdf_lo

        col_sum = P[:, j].sum()
        if col_sum > 0.0:
            P[:, j] /= col_sum

    return P

def sigma_linear_in_logM(bin_centres_logh, sigma_low=0.3, sigma_high=0.1):
    """
    Linear sigma(logM) across the mass range.
    - At lowest bin centre -> sigma_low
    - At highest bin centre -> sigma_high
    """
    x = np.asarray(bin_centres_logh)
    x_min = x.min()
    x_max = x.max()

    if x_max == x_min:
        return np.full_like(x, 0.5 * (sigma_low + sigma_high), dtype=float)

    # Linear interpolation: sigma(x_min)=sigma_low, sigma(x_max)=sigma_high
    return sigma_low + (sigma_high - sigma_low) * (x - x_min) / (x_max - x_min)


def compute_counts_per_bin(mf, M_edges_logh, V, z_mid, omegam=None, sigma8=None):
    """
    Expected counts per *true* mass bin (no scatter).

    If omegam and sigma8 are given, update the MassFunction cosmology ONCE
    for this sample using mf.update, so all internal quantities
    (P(k), sigma(M), dndm, ...) are consistent.
    """
    M_edges_logh = np.asarray(M_edges_logh)

    # Update cosmology once per sample if requested
    if (omegam is not None) or (sigma8 is not None):
        mf.update(
            z=z_mid,
            sigma_8=sigma8,
            cosmo_params={"H0": H0, "Om0": omegam},
            Mmin=M_edges_logh[0],
            Mmax=M_edges_logh[-1],
        )

    N_counts = []
    for logMmin, logMmax in zip(M_edges_logh[:-1], M_edges_logh[1:]):
        # Only change the mass bin edges; cosmology and z fixed
        mf.update(Mmin=logMmin, Mmax=logMmax)
        if len(mf.m) < 2:
            N_counts.append(0.0)
            continue
        dM = np.gradient(mf.m)
        n_mid = np.sum(mf.dndm * dM)  # h^3 Mpc^-3
        N_counts.append(n_mid * V)
    return np.array(N_counts, dtype=float)


def setup(options):
    # ----- Fiducial cosmology for MOCK generation only -----
    Om0_fid = 0.318
    sigma8_fid = 0.8

    # Redshift bin
    zmin = options.get_double(option_section, "z_min", default=0.3)
    zmax = options.get_double(option_section, "z_max", default=0.8)

    # Survey area
    area_deg2 = options.get_double(option_section, "area_deg2", default=1000.0)

    # Mass range
    mmin = options.get_double(option_section, "mass_min", default=1e14)
    mmax = options.get_double(option_section, "mass_max", default=1e15)
    if not (mmax > mmin > 0):
        raise ValueError("mass_min/mass_max must be positive and mass_max > mass_min")

    z_mid = 0.5 * (zmin + zmax)


    # Build ONE MassFunction to set up mass grid and fiducial cosmology
    mf = MassFunction(
        z=z_mid,                # will be updated in execute for each sample
        sigma_8=sigma8_fid,     # fixed fiducial for mock generation
        cosmo_params={"H0": H0, "Om0": Om0_fid},
        Mmin=np.log10(mmin),
        Mmax=np.log10(mmax),
        dlog10m=0.1,
        hmf_model="Tinker08",
    )

    # Mass bins (log10 M/h)
    nM = 10
    M_edges_logh = np.linspace(np.log10(mmin), np.log10(mmax), nM + 1)

    # Volume of the shell for mock generation (fiducial Om0, sigma8)
    V_fid = volume_shell(zmin, zmax, Om0_fid, area_deg2)

    # True (no-scatter) counts per true mass bin for fiducial cosmology
    N_true_fid = compute_counts_per_bin(mf, M_edges_logh, V_fid, z_mid)

    # Mass scatter in log10(M)
    bin_centres = 0.5 * (M_edges_logh[1:] + M_edges_logh[:-1])

    # simple mass-dependent sigma: 0.3 (low mass) -> 0.1 (high mass)
    sigma_low  = options.get_double(option_section, "sigma_low", default=0.3)
    sigma_high = options.get_double(option_section, "sigma_high", default=0.1)
    sigma_logM_data = sigma_linear_in_logM(bin_centres, sigma_low=sigma_low, sigma_high=sigma_high)

    # Build migration matrix ONCE and use it for both data and model
    P = build_migration_matrix(M_edges_logh, sigma_logM_data)

    # "Observed" (scattered) mock data vector
    N_obs_bins = P @ N_true_fid

    # For reference / possible Gaussian approximation
    N_obs_bins = np.clip(N_obs_bins, np.finfo(float).eps, None)
    sigma_obs = np.sqrt(N_obs_bins)

    config = {
        "zmin": zmin,
        "zmax": zmax,
        "area_deg2": area_deg2,
        "M_edges_logh": M_edges_logh,
        "mf": mf,
        "N_obs": N_obs_bins,   # scattered mock data
        "sigma_obs": sigma_obs,
        "sigma_logM_data": sigma_logM_data,
        "P": P,                 # SAME migration matrix used in execute
    }

    return config


def execute(block, config):
    omegam = block[names.cosmological_parameters, "omega_m"]
    sigma8 = block[names.cosmological_parameters, "sigma8_input"]

    zmin = config["zmin"]
    zmax = config["zmax"]
    area_deg2 = config["area_deg2"]
    mf = config["mf"]
    M_edges_logh = config["M_edges_logh"]
    N_obs = config["N_obs"]
    P = config["P"]  # same scatter matrix as used in setup

    # Volume for this cosmology
    V = V_shell_cached(zmin, zmax, omegam, area_deg2)
    z_mid = 0.5 * (zmin + zmax)

    # 1) True model counts (no scatter) for current (omegam, sigma8)
    N_true_model = compute_counts_per_bin(
        mf, M_edges_logh, V, z_mid, omegam=omegam, sigma8=sigma8
    )

    # 2) Apply SAME migration matrix to the MODEL as to the data
    N_model = P @ N_true_model

    # 3) Poisson likelihood
    N_model_safe = np.clip(N_model, 1e-12, None)
    loglike = np.sum(
        N_obs * np.log(N_model_safe) - N_model_safe - gammaln(N_obs + 1.0)
    )

    block["likelihoods", "hmf_like"] = loglike
    return 0


def cleanup(config):
    return 0
