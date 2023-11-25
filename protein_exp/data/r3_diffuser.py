"""R^3 diffusion methods."""
import numpy as np
from scipy.special import gamma
import torch
from data import so3_utils


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b

    def _scale(self, x):
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t):
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t*(self.max_b - self.min_b)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t) * x

    def sample_ref(self, n_samples: float=1):
        return np.random.normal(size=(n_samples, 3))

    def marginal_b_t(self, t):
        return t*self.min_b + (1/2)*(t**2)*(self.max_b-self.min_b)

    def calc_trans_0(self, score_t, x_t, t, use_torch=True):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: np.ndarray, t: float, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t).to(x_t_1.device)
        z_t_1 = torch.tensor(np.random.normal(size=x_t_1.shape)).to(x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: np.ndarray, t: float):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_0 = self._scale(x_0)
        x_t = np.random.normal(
            loc=np.exp(-1/2*self.marginal_b_t(t)) * x_0,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t)))
        )
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: float):
        return 1 / np.sqrt(self.conditional_var(t))

    def reverse(
            self,
            *,
            x_t: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
            mask: np.ndarray=None,
            center: bool=True,
            noise_scale: float=1.0,
            return_log_p_sample=False,
            use_sde: bool=True,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * torch.randn(size=score_t.shape, device=x_t.device)
        if use_sde:
            perturb_mean = (f_t - g_t**2 * score_t) * dt
            perturb = perturb_mean + g_t * np.sqrt(dt) * z
        else:
            perturb = (f_t - 0.5*g_t**2 * score_t) * dt
        
        if mask is None:
            mask = torch.ones(x_t.shape[:-1], dtype=x_t.dtype, device=x_t.device)
        perturb *= mask[..., None]
        x_t_1 = x_t - perturb
        if center:
            com = torch.sum(x_t_1, dim=-2) / torch.sum(mask, dim=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1_unscaled = self._unscale(x_t_1)
        if not return_log_p_sample:
            return x_t_1_unscaled
        
        if use_sde:
            var = (g_t * np.sqrt(dt))**2
            log_p_sample = so3_utils.normal_log_density(
                x_t_1, x_t - perturb_mean, var).sum(dim=-1)
        else:
            log_p_sample = torch.zeros_like(x_t[...,0])
        return x_t_1_unscaled, log_p_sample

    def reverse_log_prob(
            self,
            *,
            x_t: np.ndarray,
            x_tm1: np.ndarray,
            score_t: np.ndarray,
            t: float,
            dt: float,
        ):
        """Computes log probability for one step of SDE

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            x_tm1: [..., 3] positions at time t - 1  in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        # Scale down x_t and x_tm1
        x_t = self._scale(x_t)
        x_tm1 = self._scale(x_tm1)


        if not np.isscalar(t):
            raise ValueError(f'{t} must be a scalar.')
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)

        perturb_mean = (f_t - g_t**2 * score_t) * dt

        var = (g_t * np.sqrt(dt))**2
        log_p_sample = so3_utils.normal_log_density(
            x_tm1, x_t - perturb_mean, var).sum(dim=-1)
        assert len(log_p_sample.shape) == 1, log_p_sample.shape
        return log_p_sample

    def conditional_var(self, t, use_torch=False):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        if use_torch:
            return 1 - torch.exp(-self.marginal_b_t(t))
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, use_torch=False, scale=False):
        if use_torch:
            exp_fn = torch.exp
        else:
            exp_fn = np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(x_t - exp_fn(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t, use_torch=use_torch)
