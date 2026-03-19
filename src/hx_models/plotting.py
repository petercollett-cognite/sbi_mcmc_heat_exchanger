"""
Publication-quality plotting functions for heat exchanger posterior diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import ternary
from ternary.helpers import project_sequence

from hx_models.style import (
    MCMC_COLOR, SBI_COLOR, TRUE_COLOR, FOULING_COLOR, LEAK_COLOR,
)


def hdi_bands(samples_2d: np.ndarray, prob: float = 0.95):
    """Percentile-based credible bands for a (S, T) array of posterior time series."""
    if samples_2d.ndim != 2:
        raise ValueError(f"Expected (S, T) array, got shape {samples_2d.shape}")
    alpha = 1.0 - prob
    lo = np.percentile(samples_2d, 100 * alpha / 2, axis=0)
    hi = np.percentile(samples_2d, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# Failure-mode posteriors
# ---------------------------------------------------------------------------

def plot_failure_mode_categorical4_bars(
    z_mode_samples: np.ndarray,
    title: str = r"Failure Mode Posterior ($z$)",
    true_mode: int | None = None,
    labels: tuple[str, str, str, str] = ("none", "fouling", "leakage", "both"),
    savepath: str | None = None,
    ax=None,
):
    """Bar chart of the categorical4 failure-mode posterior.

    If *ax* is provided, draws on that axes (useful for subplots).
    """
    z = np.asarray(z_mode_samples).astype(int).ravel()
    probs = np.array([(z == k).mean() for k in range(4)], dtype=float)

    from hx_models.style import MODE_BAR_COLORS

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    xs = np.arange(4)
    bars = ax.bar(xs, probs, color=MODE_BAR_COLORS, alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Posterior probability")
    ax.set_title(title, fontsize=14, pad=12)
    ax.grid(True, axis="y", alpha=0.3)

    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.02, f"{p:.3f}",
                ha="center", va="bottom", fontsize=10)

    if true_mode is not None:
        from hx_models.style import TRUE_COLOR
        star_y = probs[true_mode] + 0.15
        ax.scatter([true_mode], [star_y], color=TRUE_COLOR, marker="*",
                   s=150, zorder=10, label="True mode")
        ax.set_ylim(0, max(1.1, star_y + 0.08))
        ax.legend(loc="upper right")

    if standalone:
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, bbox_inches="tight")
        plt.show()
    return probs


def plot_failure_modes_ternary(
    post_p: np.ndarray,
    n: int = 200,
    levels: int = 100,
    cmap: str = "Blues",
    title: str = "Failure Mode Posterior",
    labels: tuple[str, str, str] = ("no failure", "fouling", "leakage"),
    savepath: str | None = None,
    figsize=(6, 6),
    true_soft_mode: np.ndarray | tuple[float, float, float] | None = None,
    true_label: str = "true $p$",
    true_size: float = 60,
    true_kwargs: dict | None = None,
    show_samples: bool = True,
    sample_alpha: float = 0.15,
    n_samples_show: int = 2000,
    sample_size: float = 10.0,
    discrete_mode: bool | None = None,
    show_kde: bool = True,
    kde_jitter: float = 0.0,
    annotation_text: str | None = None,
    annotation_xy: tuple[float, float] = (0.02, 0.02),
    ax=None,
):
    """Ternary simplex plot of the Dirichlet failure-mode posterior.

    If *ax* is provided, the ternary plot is drawn on that axes (useful for
    subplots).  Otherwise a new figure is created.
    """
    post_p = np.asarray(post_p)
    if post_p.ndim != 2 or post_p.shape[1] != 3:
        raise ValueError(f"post_p must have shape (S, 3), got {post_p.shape}")

    if ax is not None:
        tax = ternary.TernaryAxesSubplot(ax=ax, scale=1.0)
        fig = ax.get_figure()
    else:
        fig, tax = ternary.figure(scale=1.0)
        fig.set_size_inches(*figsize)

    xs_samp, ys_samp = project_sequence(post_p, permutation=(0, 1, 2))
    xs_arr = np.asarray(xs_samp, dtype=float)
    ys_arr = np.asarray(ys_samp, dtype=float)

    if kde_jitter and kde_jitter > 0:
        xs_kde = xs_arr + np.random.normal(scale=kde_jitter, size=xs_arr.shape)
        ys_kde = ys_arr + np.random.normal(scale=kde_jitter, size=ys_arr.shape)
    else:
        xs_kde, ys_kde = xs_arr, ys_arr

    kde = None
    if show_kde:
        try:
            kde = gaussian_kde(np.vstack([xs_kde, ys_kde]))
        except Exception:
            kde = None

    p0 = np.linspace(0, 1, n)
    p1 = np.linspace(0, 1, n)
    P0, P1 = np.meshgrid(p0, p1)
    P2 = 1.0 - P0 - P1
    mask = (P0 >= 0) & (P1 >= 0) & (P2 >= 0)

    grid_bary = np.column_stack([P0[mask], P1[mask], P2[mask]])
    xs_grid, ys_grid = project_sequence(grid_bary, permutation=(0, 1, 2))
    ax = tax.get_axes()
    if kde is not None:
        dens_grid = kde(np.vstack([xs_grid, ys_grid]))
        ax.tricontourf(xs_grid, ys_grid, dens_grid, levels=levels,
                        cmap=cmap, alpha=0.7)

    if show_samples:
        if discrete_mode is None:
            try:
                discrete_mode = np.unique(post_p, axis=0).shape[0] <= 10
            except Exception:
                discrete_mode = False

        if discrete_mode:
            uniq_p, counts = np.unique(post_p, axis=0, return_counts=True)
            xs_u, ys_u = project_sequence(uniq_p, permutation=(0, 1, 2))
            xs_u = np.asarray(xs_u, dtype=float)
            ys_u = np.asarray(ys_u, dtype=float)
            w = counts / counts.sum()
            sizes = sample_size * 80.0 * (0.2 + 2.5 * w)
            ax.scatter(xs_u, ys_u, c=SBI_COLOR, s=sizes,
                       alpha=min(1.0, sample_alpha * 2.0), zorder=6)
        else:
            idx = np.random.choice(len(xs_arr),
                                   min(n_samples_show, len(xs_arr)), replace=False)
            ax.scatter(xs_arr[idx], ys_arr[idx], c=SBI_COLOR,
                       s=sample_size, alpha=sample_alpha, zorder=5)

    tax.boundary(linewidth=1.5)
    tax.gridlines(multiple=0.1, color="grey")
    tax.ticks(axis="lbr", multiple=0.2, linewidth=1, offset=0.02,
              tick_formats="%.1f", fontsize=8)
    tax.clear_matplotlib_ticks()
    ax.set_axis_off()
    tax.bottom_axis_label(labels[0], fontsize=12, offset=0.05)
    tax.right_axis_label(labels[1], fontsize=12, offset=0.1)
    tax.left_axis_label(labels[2], fontsize=12, offset=0.1)
    tax.set_title(title, fontsize=14, pad=12)

    if annotation_text:
        ax.text(annotation_xy[0], annotation_xy[1], annotation_text,
                transform=ax.transAxes, fontsize=10, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="black"),
                zorder=20)

    if true_soft_mode is not None:
        p_true = np.asarray(true_soft_mode, dtype=float).ravel()
        if p_true.shape != (3,):
            raise ValueError(f"true_soft_mode must be length-3, got {p_true.shape}")
        s = p_true.sum()
        if not np.isclose(s, 1.0):
            p_true = p_true / (s + 1e-12)
        x_t, y_t = project_sequence(p_true[None, :], permutation=(0, 1, 2))
        kw = dict(marker="o", s=true_size, edgecolors="black",
                  linewidths=1.0, label=true_label)
        if true_kwargs:
            kw.update(true_kwargs)
        ax.scatter(x_t, y_t, **kw)
        ax.legend(loc="upper right")

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    if ax is None:
        tax.show()
    return fig, tax


# ---------------------------------------------------------------------------
# Tau posterior
# ---------------------------------------------------------------------------

def plot_tau_est(
    posterior_or_array,
    tau_true: float | None = None,
    xlim: tuple[float, float] | None = None,
    T: int | None = None,
    show_full_prior_range: bool = True,
    title: str = r"Posterior of Changepoint $\tau$",
    savepath: str | None = None,
):
    """KDE plot of the changepoint tau posterior."""
    if isinstance(posterior_or_array, dict):
        tau = np.asarray(posterior_or_array["tau"]).ravel()
    else:
        tau = np.asarray(posterior_or_array).ravel()

    if xlim is not None:
        x_min, x_max = xlim
    elif show_full_prior_range and T is not None:
        x_min, x_max = 1.0, T - 1.0
    else:
        tau_range = np.ptp(tau)
        x_min = tau.min() - 0.1 * (tau_range + 1e-9)
        x_max = tau.max() + 0.1 * (tau_range + 1e-9)

    x_plot = np.linspace(x_min, x_max, 500)
    kde = gaussian_kde(tau)

    plt.figure(figsize=(8, 4))
    y = kde(x_plot)
    plt.plot(x_plot, y, label=r"Posterior $\tau$", linewidth=2)
    plt.fill_between(x_plot, y, alpha=0.3)
    if tau_true is not None:
        plt.axvline(tau_true, ls="--", color=TRUE_COLOR, linewidth=2,
                    label=rf"True $\tau$ = {tau_true}")
    plt.xlim(x_min, x_max)
    plt.xlabel(r"Changepoint $\tau$", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.ylim(0, None)
    plt.legend(fontsize=11)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# SBI vs MCMC comparison
# ---------------------------------------------------------------------------

def plot_sbi_mcmc_comparison(
    sbi_tau_samples: np.ndarray,
    sbi_z_mode_samples: np.ndarray,
    mcmc_tau_samples: np.ndarray,
    mcmc_z_mode_samples: np.ndarray,
    true_tau: float,
    true_z_mode: int,
    scenario_name: str = "",
    mode_names: tuple[str, ...] = ("none", "fouling", "leak", "both"),
    savepath: str | None = None,
):
    """Side-by-side tau histogram and failure-mode bar chart for SBI vs MCMC."""

    K = len(mode_names)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(sbi_tau_samples, bins=50, alpha=0.5, density=True,
                 label="SBI", color=SBI_COLOR)
    axes[0].hist(mcmc_tau_samples, bins=50, alpha=0.5, density=True,
                 label="MCMC", color=MCMC_COLOR)
    axes[0].axvline(true_tau, color=TRUE_COLOR, linestyle="--", linewidth=2,
                    label=rf"True $\tau$={true_tau:.1f}")
    axes[0].set_xlabel(r"Changepoint $\tau$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\tau$ Posterior: SBI vs MCMC")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    x_pos = np.arange(K)
    width = 0.35
    sbi_z = np.asarray(sbi_z_mode_samples).astype(int).ravel()
    mcmc_z = np.asarray(mcmc_z_mode_samples).astype(int).ravel()
    sbi_counts = np.array([(sbi_z == m).sum() for m in range(K)]) / len(sbi_z)
    mcmc_counts = np.array([(mcmc_z == m).sum() for m in range(K)]) / len(mcmc_z)

    axes[1].bar(x_pos - width/2, sbi_counts, width, label="SBI",
                alpha=0.8, color=SBI_COLOR)
    axes[1].bar(x_pos + width/2, mcmc_counts, width, label="MCMC",
                alpha=0.8, color=MCMC_COLOR)

    true_onehot = np.zeros(K)
    true_onehot[true_z_mode] = 1.0
    axes[1].scatter(x_pos, true_onehot, color=TRUE_COLOR, s=150,
                    marker="*", zorder=10, label="True")

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(mode_names)
    axes[1].set_ylabel("Posterior Probability")
    axes[1].set_title("Failure Mode $z$: SBI vs MCMC")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    if scenario_name:
        plt.suptitle(f"SBI vs MCMC -- {scenario_name}")
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()

    summary = {
        "sbi_tau_mean": float(sbi_tau_samples.mean()),
        "mcmc_tau_mean": float(mcmc_tau_samples.mean()),
        "true_tau": true_tau,
        "sbi_mode_probs": sbi_counts,
        "mcmc_mode_probs": mcmc_counts,
        "sbi_dominant_mode": int(np.argmax(sbi_counts)),
        "mcmc_dominant_mode": int(np.argmax(mcmc_counts)),
        "true_z_mode": true_z_mode,
        "sbi_correct": int(np.argmax(sbi_counts)) == true_z_mode,
        "mcmc_correct": int(np.argmax(mcmc_counts)) == true_z_mode,
    }
    return fig, axes, summary


# ---------------------------------------------------------------------------
# Fouling / leak time series posteriors
# ---------------------------------------------------------------------------

def plot_fouling_leak_time_separate(
    posterior: dict,
    true_fouling: np.ndarray | None,
    true_leak: np.ndarray | None,
    tau_true: float | None = None,
    prob_main: float = 0.68,
    prob_wide: float = 0.95,
    title: str = r"Posterior $R(t)$ and $L(t)$ Over Time",
    savepath: str | None = None,
    fouling_ylim: tuple[float, float] | None = None,
    leak_ylim: tuple[float, float] | None = None,
):
    """Two-panel plot of fouling resistance R(t) and leak fraction L(t) with HDI bands."""
    fouling_samps = np.asarray(posterior["fouling_F_t"])
    leak_samps = np.asarray(posterior["leak_frac_t"])

    median_fouling = np.median(fouling_samps, axis=0)
    median_leak = np.median(leak_samps, axis=0)
    f_lo, f_hi = hdi_bands(fouling_samps, prob=prob_main)
    f_lo95, f_hi95 = hdi_bands(fouling_samps, prob=prob_wide)
    l_lo, l_hi = hdi_bands(leak_samps, prob=prob_main)
    l_lo95, l_hi95 = hdi_bands(leak_samps, prob=prob_wide)

    T_idx = np.arange(median_fouling.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(T_idx, median_fouling, color=FOULING_COLOR, linestyle="--",
                 label="Median")
    axes[0].fill_between(T_idx, f_lo, f_hi, color=FOULING_COLOR, alpha=0.3,
                         label=f"{int(prob_main*100)}% CI")
    axes[0].fill_between(T_idx, f_lo95, f_hi95, color=FOULING_COLOR, alpha=0.1,
                         label=f"{int(prob_wide*100)}% CI")
    if true_fouling is not None:
        axes[0].plot(T_idx, true_fouling, color=TRUE_COLOR, linestyle="-",
                     linewidth=2, label="True")
    if tau_true is not None:
        axes[0].axvline(tau_true, color="black", linestyle="--",
                        label=rf"True $\tau$ = {tau_true}")
    axes[0].set_title(r"Fouling Resistance $R(t)$")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$R(t)$")
    if fouling_ylim is not None:
        axes[0].set_ylim(fouling_ylim)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(T_idx, median_leak, color=LEAK_COLOR, linestyle="--",
                 label="Median")
    axes[1].fill_between(T_idx, l_lo, l_hi, color=LEAK_COLOR, alpha=0.3,
                         label=f"{int(prob_main*100)}% CI")
    axes[1].fill_between(T_idx, l_lo95, l_hi95, color=LEAK_COLOR, alpha=0.1,
                         label=f"{int(prob_wide*100)}% CI")
    if true_leak is not None:
        axes[1].plot(T_idx, true_leak, color=TRUE_COLOR, linestyle="-",
                     linewidth=2, label="True")
    if tau_true is not None:
        axes[1].axvline(tau_true, color="black", linestyle="--",
                        label=rf"True $\tau$ = {tau_true}")
    axes[1].set_title(r"Leak Fraction $L(t)$")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$L(t)$")
    if leak_ylim is not None:
        axes[1].set_ylim(leak_ylim)
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()
    return fig, axes


def plot_fouling_leak_time(
    posterior: dict,
    true_fouling: np.ndarray | None,
    true_leak: np.ndarray | None,
    tau_true: float | None = None,
    prob_main: float = 0.68,
    prob_wide: float = 0.95,
    title: str = r"Posterior $R(t)$ and $L(t)$ Over Time (dual axis)",
    savepath: str | None = None,
):
    """Dual-axis plot of fouling R(t) and leak L(t) with HDI bands."""
    fouling_samps = np.asarray(posterior["fouling_F_t"])
    leak_samps = np.asarray(posterior["leak_frac_t"])

    median_fouling = np.median(fouling_samps, axis=0)
    median_leak = np.median(leak_samps, axis=0)
    f_lo, f_hi = hdi_bands(fouling_samps, prob=prob_main)
    f_lo95, f_hi95 = hdi_bands(fouling_samps, prob=prob_wide)
    l_lo, l_hi = hdi_bands(leak_samps, prob=prob_main)
    l_lo95, l_hi95 = hdi_bands(leak_samps, prob=prob_wide)

    T_idx = np.arange(median_fouling.shape[0])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = FOULING_COLOR
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"Fouling Resistance $R(t)$", color=color1)
    ax1.plot(T_idx, median_fouling, color=color1, linewidth=2,
             linestyle="--", label="Median $R(t)$")
    ax1.fill_between(T_idx, f_lo, f_hi, color=color1, alpha=0.3,
                     label=f"{int(prob_main*100)}% CI")
    ax1.fill_between(T_idx, f_lo95, f_hi95, color=color1, alpha=0.1,
                     label=f"{int(prob_wide*100)}% CI")
    if true_fouling is not None:
        ax1.plot(true_fouling, color=color1, linestyle="-", linewidth=2,
                 label="True $R(t)$")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = LEAK_COLOR
    ax2.set_ylabel(r"Leak Fraction $L(t)$", color=color2)
    ax2.plot(T_idx, median_leak, color=color2, linewidth=2,
             linestyle="--", label="Median $L(t)$")
    ax2.fill_between(T_idx, l_lo, l_hi, color=color2, alpha=0.3,
                     label=f"{int(prob_main*100)}% CI")
    ax2.fill_between(T_idx, l_lo95, l_hi95, color=color2, alpha=0.1,
                     label=f"{int(prob_wide*100)}% CI")
    if true_leak is not None:
        ax2.plot(true_leak, color=color2, linestyle="-", linewidth=2,
                 label="True $L(t)$")
    ax2.tick_params(axis="y", labelcolor=color2)

    if tau_true is not None:
        ax1.axvline(tau_true, color="black", linestyle="-",
                    label=rf"True $\tau$ = {tau_true}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    plt.show()
    return fig, (ax1, ax2)
