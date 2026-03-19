"""
Comparison metrics for posterior evaluation.

Metrics for continuous parameters (tau, beta_f, beta_l, lambda_rate):
- CRPS: Continuous Ranked Probability Score
- Coverage: Credible interval coverage check
- Wasserstein: Earth mover's distance between posteriors
- KL Divergence: Information-theoretic divergence (via KDE)

Metrics for categorical parameters (z_mode):
- Brier Score: Proper scoring rule for categorical predictions
- Total Variation Distance
- KL Divergence for discrete distributions
"""

import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde, entropy
from properscoring import crps_ensemble


# =============================================================================
# Continuous Parameter Metrics
# =============================================================================

def compute_crps(samples: np.ndarray, true_value: float) -> float:
    """CRPS (lower is better) for a set of posterior samples vs a true value."""
    samples = np.asarray(samples).ravel()
    return crps_ensemble(true_value, samples)


def coverage_check(
    samples: np.ndarray,
    true_value: float,
    levels: list = [0.5, 0.9, 0.95],
) -> dict:
    """Check if true value falls within credible intervals at specified levels."""
    samples = np.asarray(samples).ravel()
    results = {}

    for level in levels:
        alpha = (1 - level) / 2
        lower = np.percentile(samples, 100 * alpha)
        upper = np.percentile(samples, 100 * (1 - alpha))

        level_pct = int(level * 100)
        results[f"coverage_{level_pct}"] = lower <= true_value <= upper
        results[f"ci_width_{level_pct}"] = upper - lower
        results[f"ci_lower_{level_pct}"] = lower
        results[f"ci_upper_{level_pct}"] = upper

    return results


def compute_wasserstein(samples1: np.ndarray, samples2: np.ndarray) -> float:
    """Wasserstein-1 distance between two sample sets (lower = more similar)."""
    samples1 = np.asarray(samples1).ravel()
    samples2 = np.asarray(samples2).ravel()
    return wasserstein_distance(samples1, samples2)


def compute_kl_divergence(
    samples1: np.ndarray,
    samples2: np.ndarray,
    n_points: int = 1000,
) -> float:
    """KL(samples1 || samples2) via KDE."""
    samples1 = np.asarray(samples1).ravel()
    samples2 = np.asarray(samples2).ravel()

    kde1 = gaussian_kde(samples1)
    kde2 = gaussian_kde(samples2)

    x_min = min(samples1.min(), samples2.min())
    x_max = max(samples1.max(), samples2.max())
    padding = 0.1 * (x_max - x_min)
    x_grid = np.linspace(x_min - padding, x_max + padding, n_points)

    p = kde1(x_grid) + 1e-10
    q = kde2(x_grid) + 1e-10
    p = p / p.sum()
    q = q / q.sum()

    return entropy(p, q)


# =============================================================================
# Categorical Parameter Metrics
# =============================================================================

def samples_to_probs(z_mode_samples: np.ndarray, n_classes: int = 4) -> np.ndarray:
    """Convert discrete mode samples to empirical probability vector."""
    z_mode_samples = np.asarray(z_mode_samples).ravel().astype(int)
    counts = np.bincount(z_mode_samples, minlength=n_classes)
    return counts / len(z_mode_samples)


def compute_brier_score(probs: np.ndarray, true_class: int, n_classes: int = 4) -> float:
    """Multi-class Brier score (lower is better)."""
    probs = np.asarray(probs).ravel()
    one_hot = np.zeros(n_classes)
    one_hot[true_class] = 1.0
    return np.sum((probs - one_hot) ** 2)


def compute_tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation distance between two discrete distributions."""
    p = np.asarray(p).ravel()
    q = np.asarray(q).ravel()
    return 0.5 * np.sum(np.abs(p - q))


def compute_kl_categorical(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) for discrete distributions."""
    p = np.asarray(p).ravel()
    q = np.asarray(q).ravel()
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))


# =============================================================================
# Convenience aggregators
# =============================================================================

def compute_all_continuous_metrics(
    samples: np.ndarray,
    true_value: float,
    other_samples: np.ndarray = None,
    levels: list = [0.5, 0.9, 0.95],
) -> dict:
    """Compute all metrics for a continuous parameter."""
    results = {
        "crps": compute_crps(samples, true_value),
        "mean": np.mean(samples),
        "median": np.median(samples),
        "std": np.std(samples),
    }
    results.update(coverage_check(samples, true_value, levels))

    if other_samples is not None:
        results["wasserstein"] = compute_wasserstein(samples, other_samples)
        results["kl_forward"] = compute_kl_divergence(samples, other_samples)
        results["kl_reverse"] = compute_kl_divergence(other_samples, samples)

    return results


def compute_all_categorical_metrics(
    probs: np.ndarray,
    true_class: int,
    other_probs: np.ndarray = None,
    n_classes: int = 4,
) -> dict:
    """Compute all metrics for a categorical parameter."""
    probs = np.asarray(probs).ravel()

    results = {
        "brier_score": compute_brier_score(probs, true_class, n_classes),
        "predicted_class": np.argmax(probs),
        "correct": np.argmax(probs) == true_class,
        "prob_true_class": probs[true_class],
    }

    if other_probs is not None:
        other_probs = np.asarray(other_probs).ravel()
        results["tv_distance"] = compute_tv_distance(probs, other_probs)
        results["kl_forward"] = compute_kl_categorical(probs, other_probs)
        results["kl_reverse"] = compute_kl_categorical(other_probs, probs)

    return results
