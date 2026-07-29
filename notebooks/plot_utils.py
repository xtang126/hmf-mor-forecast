# plot_utils.py
"""Shared helpers for loading and plotting CosmoSIS forecast chains."""
import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

# --- Constants used everywhere -----------------------------------------
PARAM_LABELS = {
    "cosmological_parameters--omega_m":       r"\Omega_m",
    "cosmological_parameters--sigma8_input":  r"\sigma_8",
    "cosmological_parameters--h0":            r"h",
    "mor_parameters--frac_scatter":           r"\sigma_{\ln O}",
}

FIDUCIAL = {
    "cosmological_parameters--omega_m":       0.318,
    "cosmological_parameters--sigma8_input":  0.80,
    "cosmological_parameters--h0":            0.70,
}

DEFAULT_COLORS = ["#2a78d6", "#eda100", "#008300", "#4a3aa7", "#c8005c"]


# --- Chain loading ------------------------------------------------------
def read_cosmosis_chain(path):
    columns, meta = None, {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("## "):
                continue
            header = line[1:].strip()
            if columns is None:
                columns = header.split()
                continue
            if "=" in header:
                k, _, v = header.partition("=")
                meta[k.strip()] = v.strip()
    return np.loadtxt(path), columns, meta


def load_run(path, burnin_frac=0.5, label=None):
    """Read one chain file and return a burnin-cut MCSamples."""
    data, columns, meta = read_cosmosis_chain(path)
    nw = int(meta["walkers"])
    ns = data.shape[0] // nw
    chain = data.reshape(ns, nw, len(columns)).transpose(1, 0, 2)
    b = int(burnin_frac * ns)
    pnames = [c for c in columns if c not in ("prior", "post")]
    pidx = [columns.index(p) for p in pnames]
    flat = chain[:, b:, :].reshape(-1, len(columns))
    return MCSamples(
        samples=flat[:, pidx],
        names=pnames,
        labels=[PARAM_LABELS.get(p, p) for p in pnames],
        label=label,
    )


# --- Plotting -----------------------------------------------------------
def scan_triangle(scan_samples, param_names, legend_title,
                  filled=True, colors=None):
    colors = colors or DEFAULT_COLORS[: len(scan_samples)]
    markers = {p: FIDUCIAL[p] for p in param_names if p in FIDUCIAL}
    g = plots.get_subplot_plotter()
    g.triangle_plot(
        scan_samples, param_names,
        filled=filled,
        contour_colors=colors,
        markers=markers,
        legend_labels=[s.label for s in scan_samples],
        legend_loc="upper right",
    )
    g.legend.set_title(legend_title, prop={"size": 10, "weight": "bold"})
    return g


def scan_constraint_plot(
    scan_samples, config_values, plot_params, xlabel,
    xscale="linear", ref_x=None, colors=None,
):
    """Mean ± 1σ vs a grid axis, one subplot per parameter."""
    colors = colors or DEFAULT_COLORS[: len(scan_samples)]
    fig, axes = plt.subplots(
        1, len(plot_params),
        figsize=(5 * len(plot_params), 4), squeeze=False,
    )
    axes = axes[0]
    for i, p in enumerate(plot_params):
        ax = axes[i]
        means, errs = [], []
        for s in scan_samples:
            st = s.getMargeStats().parWithName(p)
            means.append(st.mean)
            errs.append(st.err)
        ax.plot(config_values, means, "-", color="0.75", lw=1, zorder=1)
        for cv, m, e, c in zip(config_values, means, errs, colors):
            ax.errorbar(cv, m, yerr=e, fmt="o", color=c, ecolor=c,
                        capsize=3, zorder=2)
        if ref_x is not None:
            ax.axvline(ref_x, color="0.6", ls="--", lw=1, label="matched")
        if p in FIDUCIAL:
            ax.axhline(FIDUCIAL[p], color="0.6", ls=":", lw=1, label="fiducial")
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(rf"${PARAM_LABELS.get(p, p)}$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, axes