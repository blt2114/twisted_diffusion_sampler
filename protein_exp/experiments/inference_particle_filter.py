# Import feynman_kac_pf from a level down
import sys
import os
import torch
import hydra
import numpy as np
import pandas as pd
import time
import tree
import shutil

from omegaconf import DictConfig, OmegaConf

from smc_utils_prot import feynman_kac_pf, smc_utils
from motif_scaffolding import save_motif_segments, twisting
from motif_scaffolding import utils as mu
from data import all_atom
from openfold.utils import rigid_utils as ru

from experiments import inference_motif_scaffolding
from experiments import utils as eu

from analysis import utils as au

def align_outputs_on_motif(sampler, rigids_motif, insert_motif_at_t0=False):
    """align_outputs_on_motif aligns the outputs of the sampler on the motif.

    if insert_motif is True, then the motif is inserted into the protein.

    """
    # align sample output on the motif
    sample_output = sampler.PF_cache["model_out"]
    all_motif_locations = sampler.PF_cache["all_motif_locations"]
    all_rots = sampler.PF_cache["all_rots"]
    psis = torch.zeros_like(rigids_motif[:, :2])
    atom37_motif = all_atom.compute_backbone(
        ru.Rigid.from_tensor_7(rigids_motif), psis)[0]

    max_log_p_idx = sample_output['max_log_p_idx'][0]
    motif_locations_b = all_motif_locations[max_log_p_idx]
    sampler.PF_cache['prot_traj'][0, 0] = au.align_on_motif_bb(
        torch.tensor(sampler.PF_cache['prot_traj'][0, 0]),
        motif_locations_b,
        atom37_motif,
        insert_motif=insert_motif_at_t0)
    if not "aux_traj" in sampler.PF_cache:
        return
    all_bb_prots = sampler.PF_cache['aux_traj']['all_bb_prots']
    all_bb_0_pred = sampler.PF_cache['aux_traj']['all_bb_0_pred']
    max_log_p_idx_by_t = sampler.PF_cache['max_log_p_idx_by_t']
    for p in range(all_bb_prots[0].shape[0]):
        max_log_p_idx = sample_output['max_log_p_idx'][p]
        for t in range(len(all_bb_prots)):
            # Instead use most likely rot and offset to align.

            # Center on motif and rotate to be on the motif
            f_idx = max_log_p_idx_by_t[t][p]
            motif_locations_tp = all_motif_locations[f_idx]

            # Since when all_motif_locations is computed, it is of length num_rots*num_offsets,
            # with blocks of num_rots for each offset, we get the index of the rotation like this.
            rot_idx = f_idx % all_rots.shape[0]
            R = all_rots[rot_idx].cpu().numpy()

            # Center so that most likely prediction on motif is at origin
            pred_bb = torch.tensor(all_bb_0_pred[t][p])
            motif_pred = torch.cat([pred_bb[st:(end+1)] for st, end in motif_locations_tp], dim=0)
            motif_pred_CA = motif_pred[1]
            motif_com = torch.mean(motif_pred_CA, dim=0).numpy()
            all_bb_0_pred[t][p] = all_bb_0_pred[t][p] - motif_com
            all_bb_prots[t][p] = all_bb_prots[t][p] - motif_com

            # Rotate so that most likely prediction on motif is aligned with motif
            all_bb_0_pred[t][p] = np.einsum('ij,bAj->bAi', R, all_bb_0_pred[t][p])
            all_bb_prots[t][p] = np.einsum('ij,bAj->bAi', R, all_bb_prots[t][p])

    sampler.PF_cache['aux_traj']['all_bb_prots'] = all_bb_prots
    sampler.PF_cache['aux_traj']['all_bb_0_pred'] = all_bb_0_pred

def save_aux_traj(sampler, sample_dir, aatype=None, diffuse_mask=None):
    """save_aux_traj saves the auxiliary trajectory of the particle filter,
     aligned to the motif (if provided).

     Args:
        diffuse_mask with shape [T, P, L]
    """
    if diffuse_mask is None:
        diffuse_mask = np.ones(all_bb_prots.shape[2])
    # Flip trajectory so that it starts from t=0.
    # This helps visualization.
    flip = lambda x: np.flip(np.stack(x), (0,))
    all_bb_prots = flip(sampler.PF_cache['aux_traj']['all_bb_prots'])
    all_bb_0_pred = flip(sampler.PF_cache['aux_traj']['all_bb_0_pred'])
    diffuse_mask = flip(diffuse_mask)

    for p in range(all_bb_0_pred.shape[1]):
        au.save_traj(
            all_bb_prots[:, p],
            all_bb_0_pred[:, p],
            aatype=aatype,
            diffuse_mask=diffuse_mask[:, p],
            output_dir=sample_dir,
            prefix="P{p:03d}_aux_".format(p=p)
        )


def init_particle_filter(sampler, motif_contig_info, P=4):
    sampler.PF_cache = {}
    motif_segments = [torch.tensor(motif_segment, dtype=torch.float64) for motif_segment in motif_contig_info['motif_segments']]
    rigids_motif = eu.remove_com_from_tensor_7(
        torch.cat([motif_segment.to(sampler.device) for motif_segment in motif_segments], dim=0))
    sampler.PF_cache["rigids_motif"] = rigids_motif

    if sampler._infer_conf.motif_scaffolding.use_contig_for_placement:
        num_DOF = sampler._infer_conf.motif_scaffolding.num_rots*sampler._infer_conf.motif_scaffolding.max_offsets
        assert num_DOF == 1, f"sampling using contig supported only for no DOF {num_DOF}"

    assert not sampler._infer_conf.motif_scaffolding.use_replacement, "replacement not supported for PF"
    # Check if number of possible rotations and translations is 1 and sample motif if so
    if sampler._infer_conf.motif_scaffolding.num_rots == 1 and sampler._infer_conf.motif_scaffolding.max_offsets == 1:
        motif_locations, length = sampler.motif_locations_and_length(
            motif_segments, motif_contig_info, P)
    else:
        motif_locations, length = None, motif_contig_info['length_fixed']
    sampler.PF_cache["motif_locations"] = motif_locations
    sampler.PF_cache["length"] = length

    # Compute vectorized function for degrees of freedom computation
    F, all_motif_locations, all_rots = twisting.motif_offsets_and_rots_vec_F(
        length, motif_segments, motif_locations=motif_locations, num_rots=sampler._infer_conf.motif_scaffolding.num_rots,
        device=sampler.device,
        max_offsets=sampler._infer_conf.motif_scaffolding.max_offsets,
        return_rots=True)
    sampler.PF_cache["F"] = F
    sampler.PF_cache["all_motif_locations"] = all_motif_locations
    sampler.PF_cache["all_rots"] = all_rots


    # Initialize sample_feats
    res_mask = np.ones([P, length])
    fixed_mask = np.zeros_like(res_mask)
    sample_feats = {
        'res_mask': res_mask,
        'seq_idx': torch.arange(1, length+1).repeat(P, 1),
        'fixed_mask': fixed_mask,
        'torsion_angles_sin_cos': np.zeros((P, length, 7, 2)),
        'sc_ca_t': np.zeros((P, length, 3)),
        'motif_locations': motif_locations, # list of length batch_size of lists of motif segment locations
        't_placeholder': torch.ones((P,)),
        'rigids_motif': rigids_motif,
    }

    # Add move to torch and GPU
    sample_feats = tree.map_structure(lambda x: x if (x is None or torch.is_tensor(x)) else torch.tensor(x), sample_feats)
    sample_feats = tree.map_structure(lambda x: x if x is None else x.to(sampler.device), sample_feats)
    sampler.PF_cache["sample_feats"] = sample_feats
    sampler.PF_cache["motif_contig_info"] = motif_contig_info


def log_and_validate_PF_output(
    sampler, log_w_trace, ess_trace, sample_id, sample_dir,
    keep_motif_seq=False, insert_motif_at_t0=False
    ):
    # Run self-consistency validation on the first particle
    sample_output = sampler.PF_cache["model_out"]
    all_motif_locations = sampler.PF_cache["all_motif_locations"]
    length = sampler.PF_cache["length"]
    rigids_motif = sampler.PF_cache["rigids_motif"]
    motif_contig_info = sampler.PF_cache["motif_contig_info"]


    ### Save log_w and ess_trace
    # Save log_w to file
    log_w_string = "\n".join(
        ",".join([f"{w}" for w in log_w.detach().numpy()])
        for log_w in log_w_trace)
    log_w_fn = os.path.join(sample_dir, f"log_w_{sample_id}.txt")
    with open(log_w_fn, "w") as f: f.write(log_w_string)

    # Save ess_trace to file
    ess_trace_string = ",".join([f"{ess:0.02f}" for ess in ess_trace.detach().numpy()])
    ess_trace_fn = os.path.join(sample_dir, f"ess_trace_{sample_id}.txt")
    with open(ess_trace_fn, "w") as f: f.write(ess_trace_string)

    # Compute / extract motif mask from output
    # TOD0: Adapt this to work with SMCDiff?
    max_log_p_idx = sample_output['max_log_p_idx'][0]
    motif_locations_b = all_motif_locations[max_log_p_idx]

    # Set positions of the motif in the sequence to the prescribed amino acids
    aatype = np.zeros([length], dtype=np.int32) # Default to Alanine
    for i, (st, end) in enumerate(motif_locations_b):
        aatype[st:end+1] = motif_contig_info['aatype'][i]


    # Save motif locations to file
    motif_segments_string = ",".join([f"{st}_{end}" for st, end in motif_locations_b])
    motif_segments_fn = os.path.join(sample_dir, f"motif_segments_{sample_id}.txt")
    with open(motif_segments_fn, "w") as f:
        f.write(motif_segments_string)

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((np.ones(length) * 100)[:, None], (1, 37))
    sample_path = os.path.join(sample_dir, 'sample')



    align_outputs_on_motif(sampler, rigids_motif,
            insert_motif_at_t0=insert_motif_at_t0)

    pdb_path = au.write_prot_to_pdb(
        sampler.PF_cache['prot_traj'][:, 0][0],
        sample_path,
        aatype=aatype,
        b_factors=b_factors
    )
    if sampler._infer_conf.aux_traj:
        T = len(sampler.PF_cache['aux_traj']['all_bb_prots'])
        P = sampler.PF_cache['prot_traj'][0].shape[0]
        diffuse_mask = np.zeros([T, P, length])
        # set motif positions to 1
        max_log_p_idx_by_t = sampler.PF_cache['max_log_p_idx_by_t']
        for p in range(P):
            max_log_p_idx = sample_output['max_log_p_idx'][p]
            for t in range(T):
                motif_locations_tp = all_motif_locations[max_log_p_idx_by_t[t][p]]
                for st, end in motif_locations_tp:
                    diffuse_mask[t, p, st:end+1] = 1
        save_aux_traj(sampler, sample_dir, aatype, diffuse_mask=diffuse_mask)

    # Run ProteinMPNN
    sc_output_dir = os.path.join(sample_dir, 'self_consistency')
    os.makedirs(sc_output_dir, exist_ok=True)
    shutil.copy(pdb_path, os.path.join(
        sc_output_dir, os.path.basename(pdb_path)))


def run_particle_filter(sampler, motif_contig_info, P=4, sample_id=0,
    keep_motif_seq=False, verbose=False, insert_motif_at_t0=False
 ):

    # First run init -- this does a few things:
    # * chooses the length of the backbones (# of residues)
    # * sets the degrees of freedom (if applicable) / motif locations
    init_particle_filter(sampler, motif_contig_info, P=P)

    # Check if we've already run this sample
    sample_dir = os.path.join(sampler._output_dir, f'length_{sampler.PF_cache["length"]}', f'sample_{sample_id}')
    if os.path.isdir(sample_dir):
        print("Skipping sample, prev sample already run", sample_id)
        return
    os.makedirs(sample_dir, exist_ok=True)


    # Set transition kernels and potential fucntion
    M = sampler.M
    G = sampler.G
    T = sampler._diff_conf.num_t
    resample = smc_utils.resampling_function(
        ess_threshold=sampler._infer_conf.particle_filtering.resample_threshold,
        verbose=verbose)

    # Run particle filter
    sampler._log.info(f'Running particle filter')
    xts, log_w, resample_indices_trace, ess_trace, log_w_trace  = feynman_kac_pf.smc_FK(
        M, G, resample, T, P, verbose=verbose)

    # Run self-consistency validation on the first particle
    log_and_validate_PF_output(sampler, log_w_trace, ess_trace, sample_id, sample_dir=sample_dir,
        keep_motif_seq=keep_motif_seq, insert_motif_at_t0=insert_motif_at_t0)



@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    sampler = inference_motif_scaffolding.Sampler(conf)
    torch.set_default_tensor_type('torch.FloatTensor')
    output_dir_stem = sampler._output_dir

    assert not sampler._infer_conf.motif_scaffolding.use_replacement, "use_replacement not implemented for particle filter"


    # Load motif test case details
    inpaint_df = pd.read_csv(sampler._infer_conf.motif_scaffolding.inpaint_cases_csv)
    contigs_by_test_case = save_motif_segments.load_contigs_by_test_case(inpaint_df)
    if sampler._infer_conf.motif_scaffolding.test_name is not None:
        test_names = [sampler._infer_conf.motif_scaffolding.test_name]
        print("running on test case: ", test_names)
    else:
        test_names = ["2KL8", "1BCF", "1PRW", "6EXZ_long"]# , "6EXZ_short", "1YCR", "5TPN", "7MRX_85"]
        test_names = ["1PRW", "1QJG", "5TRV_short"]# , "6EXZ_short", "1YCR", "5TPN", "7MRX_85"]
        #test_names = [name for name in contigs_by_test_case.keys()]

    for test_name in test_names:
        print("starting test case: ", test_name)
        motif_contig_info = contigs_by_test_case[test_name]

        sampler._output_dir = inference_motif_scaffolding.construct_output_dir(sampler, test_name, output_dir_stem)
        print("output_dir: ",sampler._output_dir)
        os.makedirs(sampler._output_dir, exist_ok=True)

        for sample in range(sampler._infer_conf.motif_scaffolding.number_of_samples):
            # Load in contig separately for each batch in order to sample a
            # different length for the motif in the case that the contig is used
            row = list(inpaint_df[inpaint_df.target==test_name].iterrows())[0][1]
            motif_contig_info = save_motif_segments.load_contig_test_case(row)

            run_particle_filter(
                sampler=sampler,
                motif_contig_info=motif_contig_info,
                P=sampler._infer_conf.particle_filtering.number_of_particles,
                sample_id=sample,
                keep_motif_seq=sampler._infer_conf.motif_scaffolding.keep_motif_seq,
                insert_motif_at_t0=sampler._infer_conf.particle_filtering.insert_motif_at_t0,
                )
        motif_segments = [
            torch.tensor(motif_segment, dtype=torch.float64)
            for motif_segment in motif_contig_info['motif_segments']
            ]
        rigids_motif = eu.remove_com_from_tensor_7(torch.cat([
            motif_segment.to(sampler.device) for motif_segment in motif_segments
            ], dim=0))
        sampler.run_self_consistency(
            sampler._output_dir,
            rigids_motif=rigids_motif,
            use_motif_seq=sampler._infer_conf.motif_scaffolding.keep_motif_seq,
            motif_contig_info=motif_contig_info,
            )

if __name__ == '__main__':
    run()
