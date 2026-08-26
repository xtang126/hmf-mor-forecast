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
         sigma_logM(M) = sigma_lnO(M) / (alpha * ln 10).
   sigma_lnO can be MASS-DEPENDENT (observable-dependent scatter), using the
   standard power-law trend (Evrard et al. 2014, arXiv:1403.1456; Mantz et
   al. 2016, arXiv:1606.03407):
         sigma_lnO(M) = frac_obs_scatter * (M / M_piv) ** gamma_scatter
   gamma_scatter = 0 recovers the original CONSTANT (observable-independent)
   scatter exactly -- this is what makes constant vs. mass-dependent a single
   merged model rather than two separate likelihood files.
4. Migration matrix  P_ij = P(observed bin i | true bin j), columns normalised.
5. Poisson likelihood between observed data counts and scattered model counts.
 
Two migration matrices are stored:
  - P_data:  the TRUTH used to generate the mock (frozen in setup)
  - P_model: rebuilt each MCMC step from whichever of
             mor_parameters.frac_scatter   (if scatter_model_free = True)
             mor_parameters.gamma_scatter  (if gamma_scatter_free = True)
             are sampled; anything not free is held fixed at its configured
             value. If NEITHER is free, P_model is built once in setup and
             reused every step (same performance as before).
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
FULL_SKY_DEG2 = 41253.0
 
 
# ---------------------------------------------------------------------------
# Comoving volume of a redshift shell (returned in (Mpc/h)^3)
# ---------------------------------------------------------------------------
def _volume_shell(zmin, zmax, Om0, area_deg2, h0):
    cosmo = FlatLambdaCDM(H0=h0 * 100.0, Om0=Om0)
    V = (cosmo.comoving_volume(zmax) - cosmo.comoving_volume(zmin)).to(u.Mpc**3).value
    V /= h0**3
    return V * (area_deg2 / FULL_SKY_DEG2)
 
 
@lru_cache(maxsize=2000)
def _volume_shell_cached(z1, z2, om_r, area_r, h0_r):
    return _volume_shell(float(z1), float(z2), float(om_r), float(area_r), float(h0_r))
 
 
def volume_shell(zmin, zmax, Om0, area_deg2, h0):
    return _volume_shell_cached(
        round(zmin, 4), round(zmax, 4),
        round(Om0, 5),  round(area_deg2, 3), round(h0, 5),
    )
 
 
# ---------------------------------------------------------------------------
# MOR: convert fractional observable scatter to sigma_log10(M)
# ---------------------------------------------------------------------------
def sigma_logM_from_MOR(
    frac_obs_scatter,
    alpha_MOR,
    bin_centres_logh=None,
    M_piv_logh=None,
    gamma_scatter=0.0,
):
    """
    Convert fractional (log-normal) observable scatter sigma_lnO into
    sigma_log10(M):
 
        sigma_log10(M) = sigma_lnO(M) / (alpha_MOR * ln 10)
 
    sigma_lnO is allowed to be MASS-DEPENDENT (the "observable-dependent"
    scatter case), using the power-law trend
 
        sigma_lnO(M) = frac_obs_scatter * (M / M_piv) ** gamma_scatter
 
    gamma_scatter = 0   -> sigma_lnO is CONSTANT (observable-independent).
                            This is the original behaviour, exactly.
    gamma_scatter != 0  -> sigma_lnO grows (gamma>0) or shrinks (gamma<0)
                            with mass -- the standard form used for e.g.
                            X-ray L-M intrinsic scatter trends.
 
    If bin_centres_logh is None, returns a scalar sigma_log10(M) (constant
    case, gamma_scatter is ignored). If bin_centres_logh is given, returns
    an array of length len(bin_centres_logh), one value per mass bin.
    """
    if alpha_MOR <= 0:
        raise ValueError("alpha_MOR must be positive")
 
    if bin_centres_logh is None:
        sigma_lnO = float(frac_obs_scatter)
    else:
        if M_piv_logh is None:
            raise ValueError("M_piv_logh is required when bin_centres_logh is given")
        x = np.asarray(bin_centres_logh, dtype=float) - float(M_piv_logh)  # log10(M/M_piv)
        sigma_lnO = float(frac_obs_scatter) * 10.0 ** (float(gamma_scatter) * x)
 
    return sigma_lnO / (alpha_MOR * np.log(10.0))
 
 
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
def compute_true_counts(mf, M_edges_logh, V, z_mid, omegam=None, sigma8=None, h0=None):
    edges = np.asarray(M_edges_logh)
 
    if (omegam is not None) or (sigma8 is not None) or (h0 is not None):
        mf.update(
            z=z_mid,
            sigma_8=sigma8,
            cosmo_params={"H0": h0 * 100.0, "Om0": omegam},
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
    h0_fid     = options.get_double(option_section, "h0_fid",     default=0.7)
 
    # --- Redshift range and binning ---
    zmin = options.get_double(option_section, "z_min", default=0.1)
    zmax = options.get_double(option_section, "z_max", default=1.0)
    n_z_bins = options.get_int(option_section, "n_z_bins", default=9)
    z_edges = np.linspace(zmin, zmax, n_z_bins + 1)
    z_mids = 0.5 * (z_edges[:-1] + z_edges[1:])
 
    # --- Survey area ---
    area_deg2 = options.get_double(option_section, "area_deg2", default=4000.0)
 
    # --- Mass range and binning (log10 Msun/h) ---
    mmin = options.get_double(option_section, "mass_min", default=2e14)
    mmax = options.get_double(option_section, "mass_max", default=5e15)
    if not (mmax > mmin > 0):
        raise ValueError("mass_min/mass_max must be positive and mass_max > mass_min")
    n_mass_bins = options.get_int(option_section, "n_mass_bins", default=10)
    M_edges_logh = np.linspace(np.log10(mmin), np.log10(mmax), n_mass_bins + 1)
    bin_centres_logh = 0.5 * (M_edges_logh[1:] + M_edges_logh[:-1])
 
    # --- MOR mass slope (fixed for now) ---
    alpha_MOR_data  = options.get_double(option_section, "alpha_MOR_data",  default=1.0)
    alpha_MOR_model = options.get_double(option_section, "alpha_MOR_model", default=alpha_MOR_data)
 
    # --- Pivot mass for the scatter mass-trend (Msun/h, linear) ---
    # Defaults to the geometric mean of the mass range.
    M_piv = options.get_double(
        option_section, "M_piv", default=float(np.sqrt(mmin * mmax))
    )
    M_piv_logh = np.log10(M_piv)
 
    # --- Data scatter (TRUTH for the mock, frozen) ---
    # frac_scatter_data = sigma_lnO amplitude at M = M_piv.
    # gamma_scatter_data = mass-trend slope: 0 -> constant (observable-
    # independent) scatter; != 0 -> observable(mass)-dependent scatter.
    frac_data = options.get_double(option_section, "frac_scatter_data", default=0.10)
    gamma_scatter_data = options.get_double(option_section, "gamma_scatter_data", default=0.0)
    sigma_logM_data = sigma_logM_from_MOR(
        frac_data, alpha_MOR_data,
        bin_centres_logh=bin_centres_logh, M_piv_logh=M_piv_logh,
        gamma_scatter=gamma_scatter_data,
    )
 
    # --- Model scatter mode ---
    #   scatter_model_free  = True  -> frac_scatter  sampled each step from
    #                                   mor_parameters.frac_scatter
    #   gamma_scatter_free  = True  -> gamma_scatter  sampled each step from
    #                                   mor_parameters.gamma_scatter
    #   Anything not free is held fixed at its *_model_fixed value below
    #   (default: same as the data/truth values, i.e. a "correctly
    #   specified" model -- set the _model_fixed options explicitly if you
    #   want to test a deliberately mis-specified scatter model instead).
    scatter_model_free = options.get_bool(option_section, "scatter_model_free", default=True)
    gamma_scatter_free = options.get_bool(option_section, "gamma_scatter_free", default=False)
 
    frac_model_fixed = options.get_double(
        option_section, "frac_scatter_model", default=frac_data
    )
    gamma_scatter_model_fixed = options.get_double(
        option_section, "gamma_scatter_model", default=gamma_scatter_data
    )
 
    # --- MassFunction: one instance, updated per sample and per z-bin ---
    mf = MassFunction(
        z=z_mids[0],
        sigma_8=sigma8_fid,
        cosmo_params={"H0": h0_fid * 100.0, "Om0": Om0_fid},
        Mmin=M_edges_logh[0],
        Mmax=M_edges_logh[-1],
        dlog10m=0.1,
        hmf_model="Tinker08",
    )
 
    # --- Mock data (TRUE scatter), true counts per (z-bin, mass-bin) ---
    N_true_fid = np.zeros((n_z_bins, n_mass_bins))
    for k in range(n_z_bins):
        V_k = volume_shell(z_edges[k], z_edges[k + 1], Om0_fid, area_deg2, h0_fid)
        N_true_fid[k] = compute_true_counts(
            mf, M_edges_logh, V_k, z_mids[k],
            omegam=Om0_fid, sigma8=sigma8_fid, h0=h0_fid,
        )
    P_data = build_migration_matrix(M_edges_logh, sigma_logM_data)
    N_obs = np.clip(N_true_fid @ P_data.T, np.finfo(float).eps, None)
 
    # --- Fixed P_model, reused every step only if NOTHING is free ---
    need_rebuild_pmodel = scatter_model_free or gamma_scatter_free
    P_model_fixed = None
    if not need_rebuild_pmodel:
        sigma_logM_model = sigma_logM_from_MOR(
            frac_model_fixed, alpha_MOR_model,
            bin_centres_logh=bin_centres_logh, M_piv_logh=M_piv_logh,
            gamma_scatter=gamma_scatter_model_fixed,
        )
        P_model_fixed = build_migration_matrix(M_edges_logh, sigma_logM_model)
 
    # --- Sanity print ---
    print("[hmf_mor_forecast] ================ setup ================")
    print(f"  h0_fid              = {h0_fid:.3f}")
    print(f"  z in [{zmin:.3f}, {zmax:.3f}],  n_z_bins = {n_z_bins},  area = {area_deg2:.0f} deg^2")
    print(f"  mass in [{mmin:.2e}, {mmax:.2e}] Msun/h,  n_mass_bins = {n_mass_bins}")
    print(f"  alpha_MOR_data      = {alpha_MOR_data:.3f}")
    print(f"  alpha_MOR_model      = {alpha_MOR_model:.3f}")
    print(f"  M_piv               = {M_piv:.3e} Msun/h")
    print(f"  frac_scatter_data   = {frac_data:.3f}, gamma_scatter_data = {gamma_scatter_data:.3f}")
    print(f"    -> sigma_logM(data) range = [{sigma_logM_data.min():.4f}, {sigma_logM_data.max():.4f}] dex")
    print(f"  scatter_model_free  = {scatter_model_free}   gamma_scatter_free = {gamma_scatter_free}")
    if not need_rebuild_pmodel:
        print(f"  frac_scatter_model  = {frac_model_fixed:.3f} (fixed), "
              f"gamma_scatter_model = {gamma_scatter_model_fixed:.3f} (fixed)")
        print(f"    -> sigma_logM(model) range = [{sigma_logM_model.min():.4f}, {sigma_logM_model.max():.4f}] dex")
    else:
        print("  P_model will be rebuilt every MCMC step from the free nuisance parameter(s) above.")
    print(f"  true      total counts (fid): {N_true_fid.sum():.1f}")
    print(f"  observed  total counts (fid): {N_obs.sum():.1f}")
    print("[hmf_mor_forecast] =======================================")
 
    return {
        "z_edges": z_edges,
        "z_mids": z_mids,
        "n_z_bins": n_z_bins,
        "area_deg2": area_deg2,
        "M_edges_logh": M_edges_logh,
        "bin_centres_logh": bin_centres_logh,
        "M_piv_logh": M_piv_logh,
        "mf": mf,
        "N_obs": N_obs,
        "P_data": P_data,
        "P_model_fixed": P_model_fixed,
        "alpha_MOR_data":  alpha_MOR_data,
        "alpha_MOR_model": alpha_MOR_model,
        "scatter_model_free": scatter_model_free,
        "gamma_scatter_free": gamma_scatter_free,
        "need_rebuild_pmodel": need_rebuild_pmodel,
        "frac_model_fixed": frac_model_fixed,
        "gamma_scatter_model_fixed": gamma_scatter_model_fixed,
    }
 
 
# ---------------------------------------------------------------------------
# cosmosis execute
# ---------------------------------------------------------------------------
def execute(block, config):
    omegam = block[names.cosmological_parameters, "omega_m"]
    sigma8 = block[names.cosmological_parameters, "sigma8_input"]
    h0     = block[names.cosmological_parameters, "h0"]
 
    # --- Model migration matrix (rebuilt only if something is free) ---
    if config["need_rebuild_pmodel"]:
        frac_model = (
            block["mor_parameters", "frac_scatter"]
            if config["scatter_model_free"]
            else config["frac_model_fixed"]
        )
        gamma_model = (
            block["mor_parameters", "gamma_scatter"]
            if config["gamma_scatter_free"]
            else config["gamma_scatter_model_fixed"]
        )
        sig_logM_model = sigma_logM_from_MOR(
            frac_model, config["alpha_MOR_model"],
            bin_centres_logh=config["bin_centres_logh"],
            M_piv_logh=config["M_piv_logh"],
            gamma_scatter=gamma_model,
        )
        P_model = build_migration_matrix(config["M_edges_logh"], sig_logM_model)
    else:
        P_model = config["P_model_fixed"]
 
    # True counts per (z-bin, mass-bin) at this cosmology
    z_edges = config["z_edges"]
    z_mids = config["z_mids"]
    n_z_bins = config["n_z_bins"]
    M_edges_logh = config["M_edges_logh"]
 
    N_true = np.zeros((n_z_bins, M_edges_logh.size - 1))
    for k in range(n_z_bins):
        V_k = volume_shell(z_edges[k], z_edges[k + 1], omegam, config["area_deg2"], h0)
        N_true[k] = compute_true_counts(
            config["mf"], M_edges_logh, V_k, z_mids[k],
            omegam=omegam, sigma8=sigma8, h0=h0,
        )
    N_model = np.clip(N_true @ P_model.T, 1e-12, None)
 
    # Poisson log-likelihood
    N_obs = config["N_obs"]
    loglike = float(np.sum(N_obs * np.log(N_model) - N_model - gammaln(N_obs + 1.0)))
 
    block["likelihoods", "hmf_like"] = loglike
    return 0
 
 
def cleanup(config):
    return 0
 
