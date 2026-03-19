"""
Centralized plot style configuration and paper-aligned label constants.

Usage in every notebook:
    from hx_models.style import apply_paper_style, PARAM_LABELS, OBS_LABELS, ...
    apply_paper_style()

Style follows recommendations from Rajjoub (2026) for LaTeX-friendly plots:
serif fonts, Computer Modern math, Okabe-Ito colorblind-safe palette, 10 pt base.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_paper_style():
    """Apply consistent matplotlib styling for all paper figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
        'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
        'axes.formatter.use_mathtext': True,

        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,

        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,

        'figure.figsize': (14, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',

        'axes.prop_cycle': mpl.cycler('color', [
            '#0072B2', '#D55E00', '#009E73',
            '#E69F00', '#CC79A7', '#56B4E9',
        ]),
    })


# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette (consistent across all notebooks)
# ---------------------------------------------------------------------------

SCENARIO_COLORS = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']

MCMC_COLOR = '#009E73'
SBI_COLOR = '#0072B2'
TRUE_COLOR = '#D55E00'

FOULING_COLOR = '#0072B2'
LEAK_COLOR = '#D55E00'

MODE_BAR_COLORS = ['#56B4E9', '#0072B2', '#D55E00', '#CC79A7']

# ---------------------------------------------------------------------------
# Paper-aligned labels  (code variable -> descriptive label with math)
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    "tau": r"Changepoint $\tau$",
    "beta_f": r"Fouling Strength $\beta_f$",
    "beta_l": r"Leak Rate $\beta_l$",
    "lambda_rate": r"Arrival Rate $\lambda$",
}

OBS_LABELS = {
    "Th_out": r"Hot Outlet Temp. $T_{h,\mathrm{out}}$ ($^\circ$C)",
    "Tc_out": r"Cold Outlet Temp. $T_{c,\mathrm{out}}$ ($^\circ$C)",
    "m_hot_in": r"Hot Inlet Flow $\dot{m}_{\mathrm{hot,in}}$ (kg/s)",
    "m_hot_out": r"Hot Outlet Flow $\dot{m}_{\mathrm{hot,out}}$ (kg/s)",
    "fouling_F": r"Fouling Resistance $R(t)$",
    "leak_frac": r"Leak Fraction $L(t)$",
}

MODE_LABELS = ("none", "fouling", "leakage", "both")

SCENARIO_LABELS_SHORT = [
    'Normal Op.', 'Batch SD', 'Boiler FW',
    'Mild Leak', 'Severe Leak', 'No Failure',
]

# ---------------------------------------------------------------------------
# Helpers for saving figures in multiple formats
# ---------------------------------------------------------------------------

def save_fig(fig, path_stem, formats=("png", "pdf")):
    """Save a figure in multiple formats for publication.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path_stem : str or Path
        Path without extension, e.g. ``figures/wasserstein``.
    formats : tuple of str
        File extensions to save.
    """
    from pathlib import Path
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path_stem.with_suffix(f".{fmt}"), bbox_inches='tight')
