"""SE(3) diffusion methods."""
import numpy as np
from data import so3_diffuser
from data import r3_diffuser
from scipy.spatial.transform import Rotation
from openfold.utils import rigid_utils as ru
from data import utils as du
import torch
import logging

def _extract_trans_rots(rigid: ru.Rigid):
    rot = rigid.get_rots().get_rot_mats()
    tran = rigid.get_trans()
    return tran, rot

def _assemble_rigid(R, trans, device='cpu'):
    # if trans is a torch.tensor, move to cpu and make float32

    # check if type of R is torch.tensor
    if isinstance(R, torch.Tensor):
        R = R.to(device).to(torch.float64)
    else:
        R = torch.tensor(R, dtype=torch.float64, device=device)

    # check if type of trans is torch.tensor
    if isinstance(trans, torch.Tensor):
        trans = trans.to(device).to(torch.float64)
    else:
        trans = torch.tensor(trans, dtype=torch.float64, device=device)
    return ru.Rigid(rots=ru.Rotation(rot_mats=R), trans=trans)

class SE3Diffuser:

    def __init__(self, se3_conf):
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf

        self._diffuse_rot = se3_conf.diffuse_rot
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._se3_conf.r3)

    def forward_marginal(
            self,
            rigids_0: ru.Rigid,
            t: float,
            diffuse_mask: np.ndarray = None,
            as_tensor_7: bool=True,
        ):
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true.
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        trans_0, R_0 = _extract_trans_rots(rigids_0)

        if not self._diffuse_rot:
            R_t, rot_score, rot_score_scaling = (
                R_0,
                np.zeros_like(R_0),
                np.ones_like(t)
            )
        else:
            R_t, rot_score = self._so3_diffuser.forward_marginal(
                R_0, t)
            rot_score_scaling = self._so3_diffuser.score_scaling(t)

        if not self._diffuse_trans:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                np.zeros_like(trans_0),
                np.ones_like(t)
            )
        else:
            trans_t, trans_score = self._r3_diffuser.forward_marginal(
                trans_0.cpu().numpy(), t)
            trans_t = torch.tensor(trans_t, dtype=R_0.dtype, device=R_0.device)
            trans_score = torch.tensor(trans_score, dtype=R_0.dtype, device=R_0.device)

            trans_score_scaling = self._r3_diffuser.score_scaling(t)

        if diffuse_mask is not None:
            # diffuse_mask = torch.tensor(diffuse_mask).to(rot_t.device)
            R_t = self._apply_mask(
                R_t, R_0, diffuse_mask[..., None, None])
            trans_t = self._apply_mask(
                trans_t, trans_0, diffuse_mask[..., None])

            trans_score = self._apply_mask(
                trans_score,
                np.zeros_like(trans_score),
                diffuse_mask[..., None])
            rot_score = self._apply_mask(
                rot_score,
                np.zeros_like(rot_score),
                diffuse_mask[..., None, None])
        rigids_t = _assemble_rigid(R_t, trans_t, device=trans_0.device)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {
            'R_t': R_t,
            'trans_t': trans_t,
            'rigids_t': rigids_t,
            'trans_score': trans_score,
            'rot_score': rot_score,
            'trans_score_scaling': trans_score_scaling,
            'rot_score_scaling': rot_score_scaling,
        }

    def calc_trans_0(self, trans_score, trans_t, t):
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t, trans_0, t, use_torch=False, scale=True):
        return self._r3_diffuser.score(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale)

    def calc_rot_score(self, R_t, R_0, t):
        """Returns conditional score as object in tangent space at R_t"""
        return self._so3_diffuser.torch_score(R_t, R_0, t)
        #return self._so3_diffuser.torch_score(R_t.cpu(),
        #        R_0.cpu(), t.cpu())
        #return self._so3_diffuser.torch_score(R_t.get_rot_mats().cpu(),
        #       R_0.get_rot_mats().cpu(), t.cpu())

    def _apply_mask(self, x_diff, x_fixed, diff_mask):
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(self, trans_t, score_t, t, dt, mask):
        return self._r3_diffuser.distribution(
            trans_t, score_t, t, dt, mask)

    def score(self, rigid_0: ru.Rigid, rigid_t: ru.Rigid, t: float):
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        rot_score = self._so3_diffuser.score(rot_t, t)
        trans_score = self._r3_diffuser.score(tran_t, tran_0, t)

        return trans_score, rot_score

    def score_scaling(self, t):
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
            self,
            rigid_t: ru.Rigid,
            rot_score: torch.tensor,
            trans_score: torch.tensor,
            t: float,
            dt: float,
            diffuse_mask: torch.tensor = None,
            center: bool=True,
            noise_scale: float=1.0,
            return_log_p_sample=False,
            use_sde:bool=True,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t (or tensor_7)
            rot_score: [..., N, 3, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        # check if a rigid object, if not convert to one
        if not isinstance(rigid_t, ru.Rigid):
            rigid_t = ru.Rigid.from_tensor_7(rigid_t)
        trans_t, R_t = _extract_trans_rots(rigid_t)
        R_t_1 = self._so3_diffuser.reverse(
            R_t=R_t,
            score_t=rot_score,
            t=t,
            dt=dt,
            noise_scale=noise_scale,
            return_log_p_sample=return_log_p_sample,
            use_sde=use_sde,
            )
        trans_t_1 = self._r3_diffuser.reverse(
            x_t=trans_t,
            score_t=trans_score,
            t=t,
            dt=dt,
            center=center,
            noise_scale=noise_scale,
            return_log_p_sample=return_log_p_sample,
            use_sde=use_sde,
            )

        if return_log_p_sample:
            R_t_1, R_log_p = R_t_1
            trans_t_1, trans_log_p = trans_t_1
            log_p = R_log_p + trans_log_p

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(
                trans_t_1, trans_t, diffuse_mask[..., None])
            R_t_1 = self._apply_mask(
                R_t_1, R_t, diffuse_mask[..., None, None])

        rigids = _assemble_rigid(R_t_1, trans_t_1, device=trans_t.device)
        if not return_log_p_sample:
            return rigids

        if False:
            # TODO: remove this check once we're confident that the log prob is correct
            print("checking log prob for trans full rigids")
            print("Calculation of log prob")
            print("\tR_log_p ", " ".join(f"{v:0.02f}" for v in R_log_p.detach().cpu().numpy()))
            print("\ttrans_log_p"," ".join(f"{v:0.02f}" for v in trans_log_p.detach().cpu().numpy()))
            log_p_sample_recalc = self.reverse_log_prob(
                rigid_t, rigids, rot_score, trans_score, t, dt
            )
            # Confirm that the log prob is correct
            #assert torch.allclose(log_p, log_p_sample_recalc)


        return rigids, log_p

    def reverse_log_prob(
            self,
            rigid_t: ru.Rigid,
            rigid_tm1: ru.Rigid,
            rot_score: torch.tensor,
            trans_score: torch.tensor,
            t: float,
            dt: float,
        ):
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        if not isinstance(rigid_t, ru.Rigid): rigid_t = ru.Rigid.from_tensor_7(rigid_t)
        if not isinstance(rigid_tm1, ru.Rigid): rigid_tm1 = ru.Rigid.from_tensor_7(rigid_tm1)
        trans_t, R_t = _extract_trans_rots(rigid_t)
        trans_tm1, R_tm1 = _extract_trans_rots(rigid_tm1)
        R_log_p = self._so3_diffuser.reverse_log_prob(
            R_t=R_t.cpu(),
            R_tm1=R_tm1.cpu(),
            score_t=rot_score.cpu(),
            t=t,
            dt=dt,
            )
        trans_log_p = self._r3_diffuser.reverse_log_prob(
            x_t=trans_t.cpu(),
            x_tm1=trans_tm1.cpu(),
            score_t=trans_score.cpu(),
            t=t,
            dt=dt,
            )
        #print("log prob calc. in separate function")
        #print("\tR_log_p ", " ".join(f"{v:0.02f}" for v in R_log_p.detach().cpu().numpy()))
        #print("\ttrans_log_p"," ".join(f"{v:0.02f}" for v in trans_log_p.detach().cpu().numpy()))

        log_p = R_log_p + trans_log_p
        return log_p
        

    def sample_ref(
            self,
            n_samples: int,
            impute: ru.Rigid=None,
            diffuse_mask: np.ndarray=None,
            as_tensor_7: bool=False
        ):
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
        """
        if impute is not None:
            assert impute.shape[0] == n_samples
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3, 3))
            trans_impute = self._r3_diffuser._scale(trans_impute)

        if diffuse_mask is not None and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_rot) and impute is None:
            raise ValueError('Must provide imputation values.')

        if (not self._diffuse_trans) and impute is None:
            raise ValueError('Must provide imputation values.')

        rot_ref = self._so3_diffuser.sample_ref(n_samples=n_samples)

        trans_ref = self._r3_diffuser.sample_ref(n_samples=n_samples)

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(
                rot_ref, rot_impute, diffuse_mask[..., None, None])
            trans_ref = self._apply_mask(
                trans_ref, trans_impute, diffuse_mask[..., None])
        trans_ref = self._r3_diffuser._unscale(trans_ref)
        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {'rigids_t': rigids_t}
