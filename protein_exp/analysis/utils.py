import numpy as np
import os
import re
from data import protein
from data import residue_constants
from scipy.spatial.transform import Rotation
from openfold.utils import rigid_utils
import torch


CA_IDX = residue_constants.atom_order['CA']
Rigid = rigid_utils.Rigid

def align_on_motif_bb(all_bb, motif_locs, motif_all_bb, insert_motif=True):
    """align_on_motif_bb takes a tensor of backbone atom coordinates and aligns them to the motif

    Args:
        all_bb: [N, 37, 3] tensor of backbone atom coordinates (first 3 are N, CA, C)
        motif_locs: list of tuples of (start_idx, end_idx) of motif locations
        motif_all_bb: [M, 37, 3] tensor of backbone atom coordinates (first 3 are N, CA, C)
    """
    N, _, _ = all_bb.shape
    M, _, _ = motif_all_bb.shape
    
    # check that the motif is the same length as the motif locations
    assert M == sum(end - st + 1 for st, end in motif_locs), f"M: {M}, motif_locs: {motif_locs}" 

    # Concatenate the motif locations of all_bb into a size [M, 37, 3 ] tensor
    all_bb_motif_segments = torch.cat([all_bb[st:(end+1)] for st, end in motif_locs], dim=0)

    # Make sure both the motif and the concatenated motif_all_bb are centered at zero
    motif_all_bb = motif_all_bb - motif_all_bb[:, 1:2].mean(dim=-3, keepdim=True)
    all_bb_motif_segments_mean = all_bb_motif_segments[:, 1:2].mean(dim=-3, keepdim=True)
    all_bb_motif_segments = all_bb_motif_segments - all_bb_motif_segments_mean

    # Align the concatenated all_bb_motif_segments to motif_all_bb
    R = kabsch_bb(all_bb_motif_segments[None, :, :3], motif_all_bb[:, :3])

    # align all_bb to motif_all_bb
    all_bb = all_bb - all_bb_motif_segments_mean
    all_bb = torch.einsum('ij,...j->...i', R[0], all_bb)

    # Insert the motif into all_bb
    if insert_motif:
        # iterate through the motif locations
        motif_residues_so_far = 0
        for st, end in motif_locs:
            # insert the motif into all_bb
            all_bb[st:(end+1)] = motif_all_bb[motif_residues_so_far:(motif_residues_so_far + end - st + 1)]
            # update the motif_residues_so_far
            motif_residues_so_far += end - st + 1


    return all_bb


def kabsch_bb(mobile, fixed):
    """kabsch_bb performs the Kabsch algorithm to align the mobile backbone to the fixed

    The implementation is batched and vectorized using pytorch.

    Both mobile and fixed are assumed centered at zero, and only a rotation is returned.

    Args:
        mobile: [B, M, 3, 3] tensor of mobile coordinates (N, CA, C)
        fixed: [M, 3, 3] tensor of mobile coordinates (N, CA, C)

    Returns: 
        Rotation R of shape [B, 3, 3] such that R @ mobile ~= fixed
    """
    assert len(mobile.shape) == 4, f"mobile.shape: {mobile.shape}"
    assert len(fixed.shape) == 3, f"fixed.shape: {fixed.shape}"
    B, M, _, _ = mobile.shape
    mobile = mobile.reshape([B, M*3, 3])
    fixed = fixed.reshape([M*3, 3])
    return kabsch(mobile, fixed)

def kabsch(mobile, fixed):
    """
    Args:
        mobile: [B, M, 3] tensor of mobile coordinates
        fixed: [M, 3] tensor of mobile coordinates

    Returns: 
        Rotation R of shape [B, 3, 3] such that R @ mobile ~= fixed
    """
    # Compute the covariance matrix
    H = mobile.transpose(1, 2) @ fixed
    U, _, V = torch.svd(H)
    # Compute the rotation matrix
    R = V @ U.transpose(1, 2)
    # Compute the determinant to check if it is a reflection
    det = torch.det(R)
    # If it is a reflection, flip the last column
    R[:, :, -1] *= torch.sign(det).unsqueeze(-1)
    return R


def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        aatype=None,
        b_factors=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=np.int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors)


def write_prot_to_pdb(
        prot_pos: np.ndarray,
        file_path: str,
        aatype: np.ndarray=None,
        overwrite=False,
        no_indexing=False,
        b_factors=None,
    ):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path
    with open(save_path, 'w') as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                if len(b_factors.shape) == 3: # [T, L, 37]
                    b_factors_ = b_factors[t, :, :]
                else:
                    assert len(b_factors.shape) == 2 # [L, 37]
                    b_factors_ = b_factors
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, aatype=aatype, b_factors=b_factors_)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path


def rigids_to_se3_vec(frame, scale_factor=1.0):
    trans = frame[:, 4:] * scale_factor
    rotvec = Rotation.from_quat(frame[:, :4]).as_rotvec()
    se3_vec = np.concatenate([rotvec, trans], axis=-1)
    return se3_vec


def save_traj(
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        aatype: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: str,
        x0_traj_uncond: np.ndarray = None,
        prefix: str = '',
    ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
        aatype: [N, 21] amino acid probability vector trajectory.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues and 0 for motif
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, 'sample')
    prot_traj_path = os.path.join(output_dir,prefix+ 'bb_traj')
    x0_traj_path = os.path.join(output_dir, prefix+'x0_traj')

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, :, None], (1, 1, 37))
    b_factors_0 = np.tile((diffuse_mask[0] * 100)[:, None], (1, 37))

    T = bb_prot_traj.shape[0]

    sample_path = write_prot_to_pdb(
        bb_prot_traj[0],
        sample_path,
        aatype=aatype,
        b_factors=b_factors_0
    )
    aatype_traj = np.tile(aatype[None], (T, 1, 1))
    #prot_traj_path = write_prot_to_pdb(
    #    bb_prot_traj,
    #    prot_traj_path,
    #    #aatype=aatype_traj,
    #    aatype=aatype,
    #    b_factors=b_factors
    #)
    x0_traj_path = write_prot_to_pdb(
        x0_traj,
        x0_traj_path,
        #aatype=aatype_traj,
        aatype=aatype,
        b_factors=b_factors
    )

    if x0_traj_uncond is not None:
        x0_traj_uncond_path = os.path.join(output_dir, prefix + 'x0_traj_uncond')
        x0_traj_uncond_path = write_prot_to_pdb(
            x0_traj_uncond,
            x0_traj_uncond_path,
            b_factors=b_factors
        )
    return {
        'sample_path': sample_path,
        'traj_path': prot_traj_path,
        'x0_traj_path': x0_traj_path,
    }
