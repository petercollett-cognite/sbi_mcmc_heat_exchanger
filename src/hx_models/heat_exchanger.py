"""
NumPyro probabilistic model of a counterflow heat exchanger with failure modes.

Implements the effectiveness-NTU method for steady-state outlet temperatures,
coupled with stochastic degradation models for fouling (compound Poisson process)
and internal leakage. Supports two failure-mode parameterizations:

- ``"dirichlet"``: continuous simplex prior over (no-failure, fouling, leakage)
- ``"categorical4"``: discrete mode variable z in {0: none, 1: fouling, 2: leak, 3: both}
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from numpyro import handlers
from jax import jit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@jit
def deterministic_outlet_temps(Th_in, Tc_in, m_hot_out, m_cold,
                               cp_hot, cp_cold, UA):
    """Compute outlet temperatures using the effectiveness-NTU method."""
    C_hot  = jnp.maximum(m_hot_out * cp_hot, 1e-6)
    C_cold = jnp.maximum(m_cold   * cp_cold, 1e-6)
    C_min, C_max  = jnp.minimum(C_hot, C_cold), jnp.maximum(C_hot, C_cold)
    C_min = jnp.maximum(C_min, 1e-6)
    NTU,  r       = UA / C_min, C_min / C_max
    eps_same      = NTU / (1. + NTU)
    num = -jnp.expm1(-NTU * (1. - r))
    den = 1. - r * jnp.exp(-NTU * (1. - r))
    eps_diff = num / den
    eps           = jnp.where(jnp.isclose(r, 1., 1e-8), eps_same, eps_diff)

    Q      = eps * C_min * (Th_in - Tc_in)
    Th_out = Th_in - Q / C_hot
    Tc_out = Tc_in + Q / C_cold
    return Th_out, Tc_out


def temporal_model_part(
    T=20,
    p_mode="sample",
    transition_sharpness=150.0,
    hard_step=False,
    failure_mode_model: str = "dirichlet",
    cat4_probs=None,
):
    """
    Temporal degradation model (fouling + leakage) with changepoint tau.

    Parameters
    ----------
    T : int
        Number of time steps.
    p_mode : str
        ``"sample"`` | ``"no_failure"`` | ``"fouling_only"`` | ``"leakage_only"``
    transition_sharpness : float
        Sigmoid sharpness at tau (ignored when *hard_step=True*).
    hard_step : bool
        Hard step function at tau (use for data generation, not inference).
    failure_mode_model : str
        ``"dirichlet"`` or ``"categorical4"``.
    cat4_probs : array-like or None
        Prior probabilities for categorical4 modes [none, fouling, leak, both].
        Defaults to ``[0.4, 0.2, 0.2, 0.2]``.
    """
    t = jnp.arange(T, dtype=float)
    tau01 = numpyro.sample("tau01", dist.Beta(1., 1.))
    tau   = numpyro.deterministic("tau", 1. + (T - 2.) * tau01)

    if hard_step:
        after_tau = (t >= tau).astype(float)
    else:
        after_tau = jax.nn.sigmoid(transition_sharpness * (t - tau))
    numpyro.deterministic("after_tau", after_tau)

    beta_f = numpyro.sample("beta_f", dist.LogNormal(jnp.log(0.015), 1.0))
    beta_l = numpyro.sample("beta_l", dist.LogNormal(jnp.log(0.0004), 0.4))

    with numpyro.plate("time", T):
        dF = numpyro.sample("dF", dist.Exponential(1.0 / beta_f))
        dL = numpyro.sample("dL", dist.Exponential(1.0 / beta_l))

    # -----------------------------------------------------------------
    # Failure-mode parameterization
    # -----------------------------------------------------------------
    if failure_mode_model == "dirichlet":
        if p_mode == "sample":
            p = numpyro.sample("p", dist.Dirichlet(jnp.array([1.0, 1.0, 1.0])))
        elif p_mode == "no_failure":
            p = numpyro.deterministic("p", jnp.array([1.0, 0.0, 0.0]))
        elif p_mode == "fouling_only":
            p = numpyro.deterministic("p", jnp.array([0.0, 1.0, 0.0]))
        elif p_mode == "leakage_only":
            p = numpyro.deterministic("p", jnp.array([0.0, 0.0, 1.0]))
        else:
            raise ValueError(f"Unknown p_mode: {p_mode}")

        dF = dF * after_tau * p[1]
        dL = dL * after_tau * p[2]
        numpyro.deterministic("p_both", 0.0)

    elif failure_mode_model == "categorical4":
        if p_mode == "no_failure":
            z = numpyro.deterministic("z_mode", 0)
        elif p_mode == "fouling_only":
            z = numpyro.deterministic("z_mode", 1)
        elif p_mode == "leakage_only":
            z = numpyro.deterministic("z_mode", 2)
        elif p_mode == "sample":
            if cat4_probs is None:
                cat4_probs = jnp.array([0.4, 0.2, 0.2, 0.2])
            z = numpyro.sample("z_mode", dist.Categorical(
                probs=jnp.asarray(cat4_probs)))
        else:
            raise ValueError(f"Unknown p_mode: {p_mode}")

        z = jnp.asarray(z)
        g_f = jnp.where((z == 1) | (z == 3), 1.0, 0.0)
        g_l = jnp.where((z == 2) | (z == 3), 1.0, 0.0)

        dF = dF * after_tau * g_f
        dL = dL * after_tau * g_l

        both   = jnp.where(z == 3, 1.0, 0.0)
        only_f = jnp.where(z == 1, 1.0, 0.0)
        only_l = jnp.where(z == 2, 1.0, 0.0)
        none   = jnp.where(z == 0, 1.0, 0.0)

        p_plot = jnp.array([none, only_f + 0.5 * both, only_l + 0.5 * both])
        p_plot = p_plot / (p_plot.sum() + 1e-12)
        numpyro.deterministic("p", p_plot)
        numpyro.deterministic("p_both", both)

    # -----------------------------------------------------------------
    # Compound Poisson Process fouling
    # -----------------------------------------------------------------
    lambda_rate = numpyro.sample("lambda_rate", dist.LogNormal(jnp.log(2.0), 0.5))
    jump_prob = 1.0 - jnp.exp(-lambda_rate)

    with numpyro.plate("cpp_jumps", T):
        u = numpyro.sample("cpp_u", dist.Uniform(0.0, 1.0))

    soft_mask = jax.nn.sigmoid(30.0 * (jump_prob - u))
    jump_sizes = soft_mask * dF
    fouling_F = numpyro.deterministic("fouling_F_t", jnp.cumsum(jump_sizes))

    leak_frac = numpyro.deterministic("leak_frac_t", 0.95 * (1.0 - jnp.exp(-jnp.cumsum(dL))))

    return fouling_F, leak_frac


def HX_with_failure_loop(T=20,
                         p_mode="sample",
                         transition_sharpness=150.0,
                         hard_step=False,
                         failure_mode_model: str = "dirichlet",
                         cat4_probs=None,
                         T_obs=None,
                         m_hot_in_obs=None,
                         m_hot_out_obs=None,
                         cond_obs=None,
                         T_sigma=0.20, meter_sigma=0.02, cond_sigma=0.02,
                         ):
    """
    Full heat exchanger model with failure modes.

    Parameters
    ----------
    T : int
        Number of time steps.
    p_mode : str
        Failure mode handling.
    transition_sharpness : float
        Sigmoid sharpness at tau.
    hard_step : bool
        Hard step function for transition (use for data generation).
    failure_mode_model : str
        ``"dirichlet"`` or ``"categorical4"``.
    cat4_probs : array-like or None
        Prior probabilities for categorical4 modes. Defaults to [0.4, 0.2, 0.2, 0.2].
    T_obs, m_hot_in_obs, m_hot_out_obs, cond_obs : arrays or None
        Observed data for conditioning (None = prior predictive).
    T_sigma, meter_sigma, cond_sigma : float
        Observation noise standard deviations.
    """
    UA_clean = 40_000.
    cp_cold  = numpyro.deterministic("cp_cold", 1900.)
    cp_hot   = numpyro.deterministic("cp_hot",  3500.)
    Th_in    = numpyro.deterministic("Th_in",   80.)
    Tc_in    = numpyro.deterministic("Tc_in",   25.)
    m_hot_in = numpyro.deterministic("m_hot_in_true", 2.)
    m_cold   = numpyro.deterministic("m_cold",  2.)

    fouling_F, leak_frac = temporal_model_part(
        T=T,
        p_mode=p_mode,
        transition_sharpness=transition_sharpness,
        hard_step=hard_step,
        failure_mode_model=failure_mode_model,
        cat4_probs=cat4_probs,
    )

    UA_t        = UA_clean / (1. + fouling_F)
    m_hot_out_t = m_hot_in * (1. - leak_frac)

    numpyro.deterministic("UA_t", UA_t)
    numpyro.deterministic("m_hot_out_t", m_hot_out_t)

    Th_out_t, Tc_out_t = deterministic_outlet_temps(
        Th_in, Tc_in, m_hot_out_t, m_cold, cp_hot, cp_cold, UA_t)

    cond_pred = 1e-4 * cp_hot

    with numpyro.plate("obs", T):
        numpyro.sample("Th_in_obs",  dist.Normal(Th_in,  T_sigma),
                       obs=None if T_obs is None else T_obs[:, 0])
        numpyro.sample("Tc_in_obs",  dist.Normal(Tc_in,  T_sigma),
                       obs=None if T_obs is None else T_obs[:, 1])
        numpyro.sample("Th_out_obs", dist.Normal(Th_out_t,    T_sigma),
                       obs=None if T_obs is None else T_obs[:, 2])
        numpyro.sample("Tc_out_obs", dist.Normal(Tc_out_t,    T_sigma),
                       obs=None if T_obs is None else T_obs[:, 3])
        numpyro.sample("m_hot_in_obs",  dist.Normal(m_hot_in,  meter_sigma),
                       obs=None if m_hot_in_obs  is None else m_hot_in_obs)
        numpyro.sample("m_hot_out_obs", dist.Normal(m_hot_out_t, meter_sigma),
                       obs=None if m_hot_out_obs is None else m_hot_out_obs)
        numpyro.sample("cond_obs", dist.Normal(cond_pred, cond_sigma),
                       obs=None if cond_obs      is None else cond_obs)


def generate_simulation_csv(
    T: int = 50,
    tau: float = 6.0,
    p: list = None,
    lambda_rate: float = None,
    beta_f: float = None,
    beta_l: float = None,
    rng_seed: int = 42,
    output_name: str = "simulation_data",
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Generate a single simulation and save time series to CSV.

    Parameters
    ----------
    T : int
        Number of time steps.
    tau : float
        Changepoint time.
    p : list
        Failure mode probabilities ``[p_no_fail, p_fouling, p_leak]``.
    lambda_rate, beta_f, beta_l : float or None
        Physical parameters; sampled from prior if *None*.
    rng_seed : int
        Random seed.
    output_name : str
        Base filename (without extension).
    output_dir : str or None
        Output directory. Defaults to ``data/`` relative to repo root.

    Returns
    -------
    pd.DataFrame
    """
    if p is None:
        p = [0.0, 1.0, 0.0]

    from pathlib import Path
    if output_dir is None:
        output_path = Path(__file__).resolve().parents[2] / "data"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tau01 = (tau - 1.0) / (T - 2.0)
    tau01 = np.clip(tau01, 0.0, 1.0)

    rng_key = jax.random.PRNGKey(rng_seed)

    condition_data = {"tau01": tau01, "p": jnp.array(p)}
    if lambda_rate is not None:
        condition_data["lambda_rate"] = lambda_rate
    if beta_f is not None:
        condition_data["beta_f"] = beta_f
    if beta_l is not None:
        condition_data["beta_l"] = beta_l

    model = handlers.condition(HX_with_failure_loop, data=condition_data)
    predictive = Predictive(model, num_samples=1)
    sim = predictive(rng_key, T=T, hard_step=True)

    def sq(x):
        return np.asarray(x).squeeze()

    Th_in_val = float(sq(sim["Th_in"]))
    Tc_in_val = float(sq(sim["Tc_in"]))
    cp_hot_val = float(sq(sim["cp_hot"]))
    cp_cold_val = float(sq(sim["cp_cold"]))
    m_cold_val = float(sq(sim["m_cold"]))
    m_hot_in_val = float(sq(sim["m_hot_in_true"]))
    UA_t = sq(sim["UA_t"])
    m_hot_out_t = sq(sim["m_hot_out_t"])

    Th_out_true, Tc_out_true = deterministic_outlet_temps(
        Th_in_val, Tc_in_val, m_hot_out_t, m_cold_val,
        cp_hot_val, cp_cold_val, UA_t,
    )
    Th_out_true = np.asarray(Th_out_true)
    Tc_out_true = np.asarray(Tc_out_true)

    cond_true = 1e-4 * cp_hot_val

    df = pd.DataFrame({
        "time": np.arange(T),
        "fouling_F": sq(sim["fouling_F_t"]),
        "leak_frac": sq(sim["leak_frac_t"]),
        "UA_t": UA_t,
        "Th_in_true": Th_in_val * np.ones(T),
        "Tc_in_true": Tc_in_val * np.ones(T),
        "Th_out_true": Th_out_true,
        "Tc_out_true": Tc_out_true,
        "Th_in_obs": sq(sim["Th_in_obs"]),
        "Tc_in_obs": sq(sim["Tc_in_obs"]),
        "Th_out_obs": sq(sim["Th_out_obs"]),
        "Tc_out_obs": sq(sim["Tc_out_obs"]),
        "m_hot_in_true": m_hot_in_val * np.ones(T),
        "m_hot_out_true": m_hot_out_t,
        "m_hot_in_obs": sq(sim["m_hot_in_obs"]),
        "m_hot_out_obs": sq(sim["m_hot_out_obs"]),
        "cond_true": cond_true * np.ones(T),
        "cond_obs": sq(sim["cond_obs"]),
        "after_tau": sq(sim["after_tau"]),
    })

    df.to_csv(output_path / f"{output_name}.csv", index=False)

    df_obs = pd.DataFrame({
        "time": np.arange(T),
        "Th_in_obs": sq(sim["Th_in_obs"]),
        "Tc_in_obs": sq(sim["Tc_in_obs"]),
        "Th_out_obs": sq(sim["Th_out_obs"]),
        "Tc_out_obs": sq(sim["Tc_out_obs"]),
        "m_hot_in_obs": sq(sim["m_hot_in_obs"]),
        "m_hot_out_obs": sq(sim["m_hot_out_obs"]),
        "cond_obs": sq(sim["cond_obs"]),
    })
    df_obs.to_csv(output_path / f"{output_name}_obs.csv", index=False)

    tau_actual = float(sq(sim["tau"]))
    params_df = pd.DataFrame({
        "parameter": [
            "T", "tau", "tau01", "rng_seed",
            "p_no_failure", "p_fouling", "p_leakage",
            "Th_in", "Tc_in", "m_hot_in", "m_cold",
            "cp_hot", "cp_cold",
            "beta_f", "beta_l", "lambda_rate",
        ],
        "value": [
            T, tau_actual, tau01, rng_seed,
            p[0], p[1], p[2],
            Th_in_val, Tc_in_val, m_hot_in_val, m_cold_val,
            cp_hot_val, cp_cold_val,
            float(sq(sim["beta_f"])),
            float(sq(sim["beta_l"])),
            float(sq(sim["lambda_rate"])),
        ],
    })
    params_df.to_csv(output_path / f"{output_name}_params.csv", index=False)

    return df
