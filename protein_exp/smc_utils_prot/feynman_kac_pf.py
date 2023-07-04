import torch
import numpy as np
from scipy.special import softmax, logsumexp
from tqdm.notebook import trange
from smc_utils_prot.smc_utils import compute_ess_from_log_w, normalize_log_weights

def smc_FK(M, G, resample, T, P, verbose=False):
    """smc_FK is sequential Monte Carlo in the Feynman-Kac formulation.

    This is mean to exactly mirror the formulation in Chopin's book, except stated in reverse time.

    M & G have an extra input & output, "extra_vals".  This allows the possibility of saving redundant
    computation across M and G at each step.

    Args:
        M: intial/transition distribution.
            x_T, extra_vals = M(T, None, extra_vals, P=P) # Initial sample x_T ~ M_0(dt)
            x_t, extra_vals = M(t, x_{t+1}, extra_vals) if t<T
        G: log potential function
            log_w_T, extra_vals = G(T, None, x_T, extra_vals) # Initial potential
            log_w_t, extra_vals = G(t, x_{t+1}, x_t, extra_vals) # subsequent potentials
        resample: resampling function
            log_w_t, resample_indices = resample(log_w_t) # performs resampling step (or not), resets weights
            to 0 after resamplings.
        T: final time step
        P: number of particles

    Returns:
        x_ts (shape [T, P, ...]), log_w, resample_indices_trace, ess_trace
    """
    resample_indices_trace, xts = [], []
    ess_trace = []
    log_w_trace = []
    xtp1, extra_vals = None, {}
    log_wtp1 = torch.zeros([P])

    #ts = trange(T, -1, -1) if verbose else range(T, -1, -1)
    ts = range(T, -1, -1)
    for t in ts:
        # Transition kernel and reweighting
        xt, extra_vals = M(t=t, xtp1=xtp1, extra_vals=extra_vals, P=P)
        log_w = log_wtp1 + G(t=t, xtp1=xtp1, xt=xt, extra_vals=extra_vals)
        ess = compute_ess_from_log_w(log_w)
        ess_trace.append(ess)
        log_w_trace.append(log_w)


        # Resample
        assert log_w.shape == (P,), log_w.shape
        log_w = normalize_log_weights(log_w, dim=0)
        if verbose: print(f"t={t}, ESS={ess:0.2f} w:", " ".join(
                f"{v:0.03f}" for v in torch.exp(log_w).detach().numpy())
        )

        resample_indices, resampled = resample(log_w)
        if resampled:
            # Resample and reset weights
            xt = xt[resample_indices]
            log_w *= 0.

        extra_vals[('resample_idcs', t)] = resample_indices

        # Update record lists
        resample_indices_trace.append(resample_indices)
        xts.append(xt)

        xtp1, log_wtp1 = xt, log_w


    # check if xts are numpy arrays, and if so convert to torch tensors
    if isinstance(xts[0], np.ndarray):
        xts = [torch.from_numpy(xt) for xt in xts]

    # stack xts
    xts = torch.stack(xts, dim=0)

    # Stack before returning
    ess_trace = torch.stack(ess_trace, dim=0)
    log_w_trace = torch.stack(log_w_trace, dim=0)
    resample_indices_trace = torch.stack(resample_indices_trace, dim=0)

    return xts, log_w, resample_indices_trace, ess_trace, log_w_trace
