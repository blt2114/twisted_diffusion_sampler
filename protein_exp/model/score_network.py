"""Score network module."""
import torch
import copy
import math
from torch import nn
from torch.nn import functional as F
from openfold.utils.rigid_utils import Rigid, Rotation
from data import utils as du
from data import all_atom
from model import ipa_pytorch
from motif_scaffolding import twisting
import functools as fn

Tensor = torch.Tensor


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a pruespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2).to(indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        if torch.any(node_embed.isnan()):
            print("node_embed is somewhere nan in Embedder")
            import ipdb; ipdb.set_trace()

        return node_embed, edge_embed


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats, F=None,
        use_twisting=False, twist_scale=1.,
        twist_potential_rot=True,
        twist_potential_trans=True,
        twist_update_rot=True,
        twist_update_trans=True,
    ):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """
        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t'],
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]
        if torch.any(node_embed.isnan()):
            print("node_embed is somewhere nan")
            import ipdb; ipdb.set_trace()


        # If input_feats has conditioning information, update input rigids to track gradients
        if use_twisting and "rigids_motif" in input_feats:
            # Log that we are using conditioning
            Log_delta_R, delta_x = twisting.perturbations_for_grad(input_feats, self.diffuser)

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        # Psi angle prediction
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = self._apply_mask(
            model_out['psi'], gt_psi, 1 - fixed_mask[..., None])
        pred_out = {'psi_pred': psi_pred}
        pred_out['rot_score'] = model_out['rot_score']
        pred_out['trans_score'] = model_out['trans_score']
        final_rigids = Rigid(Rotation(model_out['R_final']), model_out['trans_final'])
        model_out['final_rigids'] = final_rigids

        rigids_pred = model_out['final_rigids']
        pred_out['rigids'] = rigids_pred.to_tensor_7()

        # If input_feats has conditioning information, compute conditional score
        if use_twisting:
            grad_R_log_p_motif, grad_x_log_p_motif, max_log_p_idx, twist_log_p = twisting.grad_log_lik_approx(
                R_t=input_feats['R_t'],
                R_pred=model_out['R_final'],
                trans_pred=model_out['trans_final'],
                motif_tensor_7=input_feats['rigids_motif'],
                Log_delta_R=Log_delta_R, delta_x=delta_x,
                se3_diffuser=self.diffuser,
                t=input_feats['t'],
                F=F,
                twist_scale=twist_scale,
                twist_potential_rot=twist_potential_rot,
                twist_potential_trans=twist_potential_trans,
                )
            pred_out['max_log_p_idx'] = max_log_p_idx
            pred_out['twist_log_p'] = twist_log_p

            verbose = False
            if verbose:
                # Log the mean norms of the two gradients
                grad_R_log_p_motif_norm = torch.norm(grad_R_log_p_motif, dim=[-2, -1]).mean()
                grad_x_log_p_motif_norm = torch.norm(grad_x_log_p_motif, dim=[-1]).mean()
                print("input_feats[t]: ", input_feats['t'])
                print("grad_R_log_p_motif_norm: ", grad_R_log_p_motif_norm)
                print("grad_x_log_p_motif_norm: ", grad_x_log_p_motif_norm)


                # Log the means of the unconditioanal gradients
                grad_R_uncond = pred_out['rot_score']
                grad_x_uncond = pred_out['trans_score']
                grad_R_uncond_norm = torch.norm(grad_R_uncond, dim=[-2, -1]).mean()
                grad_x_uncond_norm = torch.norm(grad_x_uncond, dim=[-1]).mean()
                print("grad_R_uncond_norm: ", grad_R_uncond_norm)
                print("grad_x_uncond_norm: ", grad_x_uncond_norm)


            # scale grad_R_log_p_motif such that each 3x3 matrix can have Frobenius norm at most 1000
            if sum(torch.isnan(grad_R_log_p_motif).flatten()) > 0:
                num_nans = sum(torch.isnan(grad_R_log_p_motif).flatten())
                print("grad_R_log_p_motif has ", num_nans, " nans")

                # set the nans to 0
                # first find indices corresponding to nans
                nan_indices = torch.where(torch.isnan(grad_R_log_p_motif[0]).sum(dim=[-2,-1]))[0]
                # set  rotation matrices to zero if they have nans
                grad_R_log_p_motif[0, nan_indices] = 0.


            # Consider doing something similar for translations? (i.e. for scaling)
            # TODO: Do ablation to check if this matters! (i.e. if we don't scale the gradients)
            max_norm = 1e3
            norms = torch.norm(grad_R_log_p_motif, dim=[-2, -1], keepdim=True) # keep the last dimensions
            if sum(norms.flatten() > max_norm) > 0:
                print("norms of grad_R_log_p_motif are ", norms.shape, norms.flatten())
            grad_R_scaling = max_norm / (max_norm + norms)
            grad_R_log_p_motif = grad_R_scaling*grad_R_log_p_motif

            if sum(norms.flatten() > max_norm) > 0:
                print("norms of grad_trans_log_p_motif are ", norms.shape, norms.flatten())
            #norms = torch.norm(grad_x_log_p_motif, dim=[-1], keepdim=True) # keep the last dimensions
            #if sum(norms.flatten() > max_norm) > 0:
            #    print("norms of grad_trans_log_p_motif are ", norms.shape, norms.flatten())
            #grad_x_scaling = max_norm / (max_norm + norms)
            #grad_x_log_p_motif = grad_x_scaling*grad_x_log_p_motif

            if twist_update_rot:
                pred_out['rot_score_uncond'] = pred_out['rot_score'].detach().clone()
                pred_out['rot_score'] = pred_out['rot_score'] + grad_R_log_p_motif
            if twist_update_trans:
                pred_out['trans_score_uncond'] = pred_out['trans_score'].detach().clone()
                pred_out['trans_score'] = pred_out['trans_score'] + grad_x_log_p_motif

            bar_a_t = torch.exp(-self.diffuser._r3_diffuser.marginal_b_t(input_feats['t']))
            factor_on_score_x = (1-bar_a_t)/torch.sqrt(bar_a_t)

            rigids_pred = Rigid.from_tensor_7(pred_out['rigids'])
            pred_out['rigids_uncond'] = pred_out['rigids'].detach().clone()
            x_pred = rigids_pred.get_trans()
            x_pred = x_pred + factor_on_score_x[:, None, None] * self.diffuser._r3_diffuser._unscale(grad_x_log_p_motif)
            rigids_pred._trans = x_pred
            pred_out['rigids'] = rigids_pred.to_tensor_7()


        for k, v in input_feats.items():
            # check if a the value is a tensor, and detach if so.
            if isinstance(v, torch.Tensor):
                input_feats[k] = v.detach()


        return pred_out
