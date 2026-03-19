"""
MCMC and SBI inference utilities for the heat exchanger model.

Provides:
- MCMC inference via NumPyro NUTS (with DiscreteHMCGibbs for categorical4)
- SBI simulation wrappers and latent-space transformations
- Observation packing / unpacking helpers
- Prior constants
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from numpyro.infer import MCMC, NUTS, Predictive, DiscreteHMCGibbs, init_to_sample
from numpyro import handlers

import numpy as np
from tqdm import tqdm

import hx_models.heat_exchanger as hx

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODE_NAMES = ["none", "fouling", "leak", "both"]
K_CATEGORICAL4 = 4

PRIOR_LOG_BETA_F_MEAN = np.log(0.015)
PRIOR_LOG_BETA_F_STD = 1.0
PRIOR_LOG_BETA_L_MEAN = np.log(0.0004)
PRIOR_LOG_BETA_L_STD = 0.4
PRIOR_LOG_LAMBDA_MEAN = np.log(2.0)
PRIOR_LOG_LAMBDA_STD = 0.5


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------

def p_to_z_mode(p):
    """Convert 3-way probability vector to categorical4 z_mode."""
    p = np.asarray(p)
    if np.allclose(p, [1, 0, 0]):
        return 0
    elif np.allclose(p, [0, 1, 0]):
        return 1
    elif np.allclose(p, [0, 0, 1]):
        return 2
    else:
        return 3


def z_mode_to_onehot(z_mode, K=K_CATEGORICAL4):
    """Convert z_mode to one-hot vector."""
    onehot = np.zeros(K)
    onehot[int(z_mode)] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# SBI latent-space transformations
# ---------------------------------------------------------------------------

def latent_to_simtheta(theta_latent: "torch.Tensor", K: int = K_CATEGORICAL4) -> "torch.Tensor":
    """Latent (tau01, logits) -> simulator (tau01, z_mode) via categorical sampling."""
    theta_latent = theta_latent.float()
    tau01 = theta_latent[:, :1]
    logits = theta_latent[:, 1:1+K]
    probs = F.softmax(logits, dim=-1)
    z_mode = torch.multinomial(probs, num_samples=1)
    return torch.cat([tau01, z_mode.float()], dim=1)


def latent_to_probs(theta_latent: "torch.Tensor", K: int = K_CATEGORICAL4) -> "torch.Tensor":
    """Latent -> (tau01, mode_probs) without sampling z_mode."""
    theta_latent = theta_latent.float()
    tau01 = theta_latent[:, :1]
    logits = theta_latent[:, 1:1+K]
    probs = F.softmax(logits, dim=-1)
    return torch.cat([tau01, probs], dim=1)


def latent_to_simtheta_extended(theta_latent: "torch.Tensor", K: int = K_CATEGORICAL4) -> "torch.Tensor":
    """Extended latent -> simulator params (tau01, z_mode, beta_f, beta_l, lambda_rate)."""
    theta_latent = theta_latent.float()

    tau01      = theta_latent[:, :1]
    logits     = theta_latent[:, 1:1+K]
    log_beta_f = theta_latent[:, 1+K:2+K]
    log_beta_l = theta_latent[:, 2+K:3+K]
    log_lambda = theta_latent[:, 3+K:4+K]

    probs = F.softmax(logits, dim=-1)
    z_mode = torch.multinomial(probs, num_samples=1).float()

    beta_f      = torch.exp(log_beta_f)
    beta_l      = torch.exp(log_beta_l)
    lambda_rate = torch.exp(log_lambda)

    return torch.cat([tau01, z_mode, beta_f, beta_l, lambda_rate], dim=1)


def latent_to_params_extended(theta_latent: "torch.Tensor", K: int = K_CATEGORICAL4) -> "torch.Tensor":
    """Extended latent -> interpretable params (tau01, probs, beta_f, beta_l, lambda_rate)."""
    theta_latent = theta_latent.float()

    tau01      = theta_latent[:, :1]
    logits     = theta_latent[:, 1:1+K]
    log_beta_f = theta_latent[:, 1+K:2+K]
    log_beta_l = theta_latent[:, 2+K:3+K]
    log_lambda = theta_latent[:, 3+K:4+K]

    probs       = F.softmax(logits, dim=-1)
    beta_f      = torch.exp(log_beta_f)
    beta_l      = torch.exp(log_beta_l)
    lambda_rate = torch.exp(log_lambda)

    return torch.cat([tau01, probs, beta_f, beta_l, lambda_rate], dim=1)


# ---------------------------------------------------------------------------
# Summary statistics for SBI
# ---------------------------------------------------------------------------

def compute_summary_statistics(x: "torch.Tensor", T: int) -> "torch.Tensor":
    """Compute informative summary statistics from raw (N, 7*T) observations."""
    N = x.shape[0]
    x = x.reshape(N, 7, T)

    Th_in    = x[:, 0, :]
    Tc_in    = x[:, 1, :]
    Th_out   = x[:, 2, :]
    Tc_out   = x[:, 3, :]
    m_hot_in = x[:, 4, :]
    m_hot_out = x[:, 5, :]

    delta_T_hot  = Th_in - Th_out
    delta_T_cold = Tc_out - Tc_in
    mass_loss    = m_hot_in - m_hot_out

    summaries = []
    for arr in [delta_T_hot, delta_T_cold, mass_loss, Th_out, Tc_out]:
        summaries.append(arr.mean(dim=1, keepdim=True))
        summaries.append(arr.std(dim=1, keepdim=True))
        q1 = arr[:, :T//4].mean(dim=1, keepdim=True)
        q4 = arr[:, -T//4:].mean(dim=1, keepdim=True)
        summaries.append(q4 - q1)
        summaries.append(arr.max(dim=1, keepdim=True)[0] - arr.min(dim=1, keepdim=True)[0])
        t_norm = torch.linspace(-1, 1, T).unsqueeze(0).expand(N, -1)
        slope = (t_norm * arr).mean(dim=1, keepdim=True) / (t_norm**2).mean()
        summaries.append(slope)

    return torch.cat(summaries, dim=1)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def pack_obs(sim):
    """Pack a NumPyro Predictive output into the dict expected by ``do_inference``."""
    def sq(x):
        x = jnp.asarray(x)
        return x.squeeze(0) if x.ndim > 1 and x.shape[0] == 1 else x

    return dict(
        T_obs         = jnp.stack([sq(sim["Th_in_obs"]), sq(sim["Tc_in_obs"]),
                                   sq(sim["Th_out_obs"]), sq(sim["Tc_out_obs"])], axis=1),
        m_hot_in_obs  = sq(sim["m_hot_in_obs"]),
        m_hot_out_obs = sq(sim["m_hot_out_obs"]),
        cond_obs      = sq(sim["cond_obs"]),
    )


def flat_to_mcmc_obs(obs_flat, T):
    """Convert a flat ``(7*T,)`` observation array to the MCMC observation dict."""
    data = np.asarray(obs_flat).reshape(7, T)
    return {
        "T_obs":         jnp.array(data[:4].T),
        "m_hot_in_obs":  jnp.array(data[4]),
        "m_hot_out_obs": jnp.array(data[5]),
        "cond_obs":      jnp.array(data[6]),
    }


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(rng_key, params, num_samples=1, T=100, failure_mode_model="dirichlet"):
    """Run simulation with conditioned parameters."""
    conditioned_model = handlers.condition(hx.HX_with_failure_loop, data=params)
    predictive = Predictive(conditioned_model, num_samples=num_samples)
    return predictive(rng_key, T=T, failure_mode_model=failure_mode_model)


def get_observation(subkey, num_samples=1):
    """Generate a prior-predictive observation."""
    predictive = Predictive(hx.HX_with_failure_loop, num_samples=num_samples)
    return predictive(subkey)


# ---------------------------------------------------------------------------
# MCMC inference
# ---------------------------------------------------------------------------

def do_inference(
    rng_key,
    obs,
    T=None,
    num_warmup=500,
    num_samples=2500,
    num_chains=4,
    failure_mode_model: str = "dirichlet",
    cat4_probs=None,
):
    """
    Run MCMC inference on observed data.

    Uses NUTS for dirichlet and DiscreteHMCGibbs(NUTS) for categorical4.

    Parameters
    ----------
    cat4_probs : array-like or None
        Prior probabilities for categorical4 modes. Defaults to [0.4, 0.2, 0.2, 0.2].
    """
    base = NUTS(hx.HX_with_failure_loop, init_strategy=init_to_sample)
    if failure_mode_model == "dirichlet":
        kernel = base
    elif failure_mode_model == "categorical4":
        kernel = DiscreteHMCGibbs(base)
    else:
        raise ValueError(f"Unknown failure_mode_model: {failure_mode_model}")

    mcmc = MCMC(kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                chain_method="parallel")

    if T is None:
        T = obs["T_obs"].shape[0]

    mcmc.run(
        rng_key,
        T=T,
        failure_mode_model=failure_mode_model,
        cat4_probs=cat4_probs,
        T_obs=obs["T_obs"],
        m_hot_in_obs=obs["m_hot_in_obs"],
        m_hot_out_obs=obs["m_hot_out_obs"],
        cond_obs=obs["cond_obs"],
    )
    return mcmc


# ---------------------------------------------------------------------------
# SBI simulation wrappers
# ---------------------------------------------------------------------------

def get_observations_parameters(params, failure_mode_model="dirichlet"):
    """Convert tensor parameters to list of dicts for simulation."""
    params_np = params.numpy() if isinstance(params, torch.Tensor) else np.asarray(params)
    params_list = []

    for i in range(params_np.shape[0]):
        if failure_mode_model == "dirichlet":
            params_list.append({
                "tau01": np.array(params_np[i, 0]),
                "p": np.array(params_np[i, 1:4]),
            })
        elif failure_mode_model == "categorical4":
            params_list.append({
                "tau01": np.array(params_np[i, 0]),
                "z_mode": int(params_np[i, 1]),
            })
        elif failure_mode_model == "categorical4_extended":
            params_list.append({
                "tau01": np.array(params_np[i, 0]),
                "z_mode": int(params_np[i, 1]),
                "beta_f": float(params_np[i, 2]),
                "beta_l": float(params_np[i, 3]),
                "lambda_rate": float(params_np[i, 4]),
            })
        else:
            raise ValueError(f"Unknown failure_mode_model: {failure_mode_model}")

    return params_list


def run_multiple_observations(rng_key, params_list, T=100, failure_mode_model="dirichlet"):
    """Run multiple simulations (with progress bar) for SBI training data."""
    results = []
    desc = f"Simulating ({failure_mode_model})"
    for d in tqdm(params_list, desc=desc):
        rng_key, subkey = jax.random.split(rng_key)
        sim = simulate(subkey, d, num_samples=1, T=T, failure_mode_model=failure_mode_model)
        results.append(sim)
    return results


def unpack_sim_for_SBI(sim):
    """Flatten a list of simulation dicts into a ``(N, 7*T)`` tensor for SBI."""
    Tc_in_obs, Th_in_obs = [], []
    Tc_out_obs, Th_out_obs = [], []
    m_hot_in_obs, m_hot_out_obs, cond_obs = [], [], []

    for s in sim:
        Th_in_obs.append(s["Th_in_obs"].squeeze(0))
        Tc_in_obs.append(s["Tc_in_obs"].squeeze(0))
        Th_out_obs.append(s["Th_out_obs"].squeeze(0))
        Tc_out_obs.append(s["Tc_out_obs"].squeeze(0))
        m_hot_in_obs.append(s["m_hot_in_obs"].squeeze(0))
        m_hot_out_obs.append(s["m_hot_out_obs"].squeeze(0))
        cond_obs.append(s["cond_obs"].squeeze(0))

    res = [Th_in_obs, Tc_in_obs, Th_out_obs, Tc_out_obs,
           m_hot_in_obs, m_hot_out_obs, cond_obs]
    res = torch.swapaxes(torch.tensor(np.array(res)), 0, 1)
    return res.reshape(res.shape[0], -1).contiguous()


def simulation_wrapper_sbi(params, T=100, failure_mode_model="dirichlet"):
    """SBI simulation wrapper: params tensor -> observations tensor."""
    rng_key = jax.random.PRNGKey(np.random.randint(0, 2**32 - 1))
    params_list = get_observations_parameters(params, failure_mode_model=failure_mode_model)
    sim_model = "categorical4" if failure_mode_model == "categorical4_extended" else failure_mode_model
    res = run_multiple_observations(rng_key, params_list, T=T, failure_mode_model=sim_model)
    return unpack_sim_for_SBI(res)


def simulation_wrapper_sbi_extended(theta_latent, T=100, K=K_CATEGORICAL4):
    """Extended SBI wrapper: latent -> simulation -> (N, 7*T) observations."""
    sim_params = latent_to_simtheta_extended(theta_latent, K)
    return simulation_wrapper_sbi(sim_params, T=T, failure_mode_model="categorical4_extended")
