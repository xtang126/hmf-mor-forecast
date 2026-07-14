# hmf-mor-forecast

Forecasting pipeline for cluster cosmology constraints from the halo mass
function (HMF), including mass–observable relation (MOR) scatter.

Built as a [CosmoSIS](https://cosmosis.readthedocs.io/) likelihood module. It
compares mock cluster counts (generated from a fiducial cosmology with the
[`hmf`](https://hmf.readthedocs.io/) package, Tinker08 mass function) against
model predictions across a grid of `(omega_m, sigma8)`, with a Gaussian
migration matrix in log-mass to account for mass–observable scatter, and
evaluates a Poisson likelihood.

## Layout

- `likelihoods/` — CosmoSIS likelihood module (`hmf_mor_forecast.py`): builds
  mock "observed" counts per mass bin, applies the mass-scatter migration
  matrix to both data and model, and computes the Poisson log-likelihood.
- `configs/` — CosmoSIS pipeline `.ini` files (sampler, priors, pipeline
  wiring) and parameter-values files.
- `scripts/` — helper / driver scripts.
- `notebooks/` — exploratory analysis and plotting notebooks.
- `figures/` — generated plots.
- `output/` — pipeline run outputs (chains, fits) — **not tracked in git**.

## Requirements

- [CosmoSIS](https://cosmosis.readthedocs.io/)
- `hmf`
- `astropy`
- `numpy`, `scipy`

## Usage

Run a forecast with CosmoSIS, pointing at one of the config files, e.g.:

```bash
cosmosis configs/forecast_M2e14_A4000_sd10_sm10.ini
```

Outputs (chains, fits) are written under `output/`.

## Contact

Xin Tang (xt52@sussex.ac.uk)
