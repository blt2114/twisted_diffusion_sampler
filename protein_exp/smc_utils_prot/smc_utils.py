import torch as th
from scipy.special import logsumexp


def compute_ess(w, dim=0):
    ess = (w.sum(dim=dim))**2 / th.sum(w**2, dim=dim)
    return ess

def compute_ess_from_log_w(log_w, dim=0):
    return compute_ess(normalize_weights(log_w, dim=dim), dim=dim)

def normalize_weights(log_weights, dim=0):
    return th.exp(normalize_log_weights(log_weights, dim=dim))

def normalize_log_weights(log_weights, dim):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - th.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights


def _resample_particles(x, log_weights, ess_threshold=None):
    # TODO: check other SMC implementation
    # x: (num_particles, batch_size, ...)
    # log_weights: (num_particles, batch_size)

    # shape check
    num_particles, batch_size = log_weights.shape
    assert x.shape[:2] == log_weights.shape, f"x.shape = {x.shape}, log_weights.shape = {log_weights.shape}"

    # normalize weights
    log_weights[th.isnan(log_weights)] = - th.inf
    w = th.softmax(log_weights - log_weights.max(dim=0, keepdims=True)[0], dim=0)
    resample = True

    ess = 1./th.sum(w**2, dim=0) # (batch_size)

    if ess_threshold is not None:
        resample = (ess < (num_particles * ess_threshold))

    print("ess", ess)
    print("resample", resample)

    # TODO: for batch_size = 1, can skip this step if not resample
    # multinomial resampling for now.
    resample_indicies = th.multinomial(w.transpose(0,1), num_samples=num_particles, replacement=True).transpose(0,1)
    assert resample_indicies.shape == (num_particles, batch_size)
    resampled_x = batch_gather(x, resample_indicies)

    _x_shape = [1] * len(x.shape[2:])
    resample = resample.view(*resample.shape, *_x_shape)  # (bsz, ... )
    resampled_x = resample * resampled_x + (~resample) * x

    # assert only when not resampling
    #assert th.equal(resampled_x, x)
    return resampled_x, ess, resample_indicies


def batch_gather(x, indices):

    # indices_shape = (num_particles, bsz)
    # x shape = (num_particles, bsz, ...)

    assert indices.shape == x.shape[:2]

    _x_shape = [1] * len(x.shape[2:])
    expanded_indices = indices.view(*indices.shape, *_x_shape)
    expanded_indices = expanded_indices.expand(x.shape)
    assert expanded_indices.shape == x.shape

    resampled_x = x.gather(0, expanded_indices)

    return resampled_x



import numpy as np
from numba import jit
from numpy import random


@jit(nopython=True)
def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution.
        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        W: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)
        Returns
        -------
        A: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            if j == M-1:
                break  # avoiding numerical issue
            j += 1
            s += W[j]
        A[n] = j
    return A

def uniform_spacings(N):
    """Generate ordered uniform variates in O(N) time.
    Parameters
    ----------
    N: int (>0)
        the expected number of uniform variates
    Returns
    -------
    (N,) float ndarray
        the N ordered variates (ascending order)
    Note
    ----
    This is equivalent to::
        from numpy import random
        u = sort(random.rand(N))
    but the line above has complexity O(N*log(N)), whereas the algorithm
    used here has complexity O(N).
    """
    z = np.cumsum(-np.log(random.rand(N + 1)))
    return z[:-1] / z[-1]


def multinomial(W, M):
    """Multinomial resampling.
    Popular resampling scheme, which amounts to sample N independently from
    the multinomial distribution that generates n with probability W^n.
    This resampling scheme is *not* recommended for various reasons; basically
    schemes like stratified / systematic / SSP tends to introduce less noise,
    and may be faster too (in particular systematic).
    """
    return inverse_cdf(uniform_spacings(M), W)


def stratified(W, M):
    """Stratified resampling.
    """
    su = (random.rand(M) + np.arange(M)) / M
    return inverse_cdf(su, W)


def systematic(W, M):
    """Systematic resampling.
    """
    su = (random.rand(1) + np.arange(M)) / M
    return inverse_cdf(su, W)


def residual(W, M):
    """Residual resampling.
    """
    N = W.shape[0]
    A = np.empty(M, dtype=np.int64)
    MW = M * W
    intpart = np.floor(MW).astype(np.int64)
    sip = np.sum(intpart)
    res = MW - intpart
    sres = M - sip
    A[:sip] = np.arange(N).repeat(intpart)
    # each particle n is repeated intpart[n] times
    if sres > 0:
        A[sip:] = multinomial(res / sres, M=sres)
    return A



Resample_dict = dict(systematic=systematic,
                     stratified=stratified,
                     residual=residual,
                     multinomial=multinomial)


def resample(weights, strategy='systematic'):
    P, B = weights.shape
    resample_indices = th.empty(P, B, dtype=th.int64)

    resample_fn = Resample_dict[strategy]
    for b in range(B):
        resample_indices_b = resample_fn(weights[:,b].cpu().numpy(), P)
        resample_indices[:,b] = th.from_numpy(resample_indices_b) #.to(normalize_weights.device)
    return resample_indices


def resample_b(weights_b, strategy='systematic'):
    P, = weights_b.shape

    resample_fn = Resample_dict[strategy]
    resample_indices_b = th.from_numpy(resample_fn(weights_b.cpu().numpy(), P))
    return resample_indices_b

def resampling_function(resample_strategy="systematic", ess_threshold=None, verbose=False):
    """resampling_function returns a resampling function that may be used in
    smc_FK
    """
    resample_fn = Resample_dict[resample_strategy]

    def resample(log_w):
        normalized_weights = normalize_weights(log_w, dim=-1)
        P = log_w.shape[0]
        ess = compute_ess(normalized_weights, dim=0)
        if  ess_threshold is None or ess < P*ess_threshold:
            if verbose: print("resample")
            resample_indices = th.from_numpy(resample_fn(W=np.array(normalized_weights.cpu()), M=P))
            is_resampled = True
        else:
            resample_indices = th.arange(P)
            is_resampled = False
        return resample_indices, is_resampled

    return resample


if __name__ == "__main__":
    def test_take_indices(x, indices):
        resampled_x = batch_gather(x, indices)

        # iterate over batch dimension
        for b in range(indices.shape[1]):
            assert th.equal(resampled_x[:,b], x[:,b][indices[:,b]])


    x = th.randn(3,2, 10, 5)

    test_take_indices(x, th.tensor([[1, 0], [2, 0], [0, 2]]))
    test_take_indices(x, th.tensor([[0,2], [1,0], [1, 1]]))
