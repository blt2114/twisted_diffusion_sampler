"""Script for running inference and sampling.

Sample command:
> python scripts/run_inference.py

"""
import torch
import sys
import copy
import os
import time
import tree
import numpy as np
import hydra
import random
import itertools
import subprocess
import logging
import pandas as pd
import traceback
import shutil
from datetime import datetime
from biotite.sequence.io import fasta
import GPUtil
from typing import Optional

from analysis import utils as au
from analysis import metrics
from data import utils as du
from data import residue_constants
from data import all_atom
from typing import Dict
from experiments import train_se3_diffusion, inference_se3_diffusion
from experiments import utils as eu
from motif_scaffolding import save_motif_segments
from motif_scaffolding import utils as mu
from omegaconf import DictConfig, OmegaConf
from openfold.data import data_transforms
from openfold.utils.rigid_utils import Rigid
from openfold.utils import rigid_utils as ru
from motif_scaffolding import twisting
import esm




CA_IDX = residue_constants.atom_order['CA']

def construct_output_dir(sampler, test_name, output_dir_stem):
    twist_scale = sampler._infer_conf.motif_scaffolding.twist_scale
    no_self_cond = sampler._infer_conf.motif_scaffolding.no_self_conditioning
    num_rots = sampler._infer_conf.motif_scaffolding.num_rots
    max_offsets = sampler._infer_conf.motif_scaffolding.max_offsets

    num_steps = sampler._infer_conf.diffusion.num_t
    noise_scale = sampler._infer_conf.diffusion.noise_scale

    min_t = sampler._infer_conf.diffusion.min_t
    twist_rot = sampler._infer_conf.motif_scaffolding.twist_potential_rot
    twist_trans = sampler._infer_conf.motif_scaffolding.twist_potential_trans
    insert_motif_at_t0 = sampler._infer_conf.particle_filtering.insert_motif_at_t0
    resample_threshold = sampler._infer_conf.particle_filtering.resample_threshold

    N_particles = sampler._infer_conf.particle_filtering.number_of_particles

    folding_method = sampler._infer_conf.folding_method

    # Name and create output directory
    name_segments = [test_name + "/"]

    if sampler._infer_conf.motif_scaffolding.use_twisting:
        name_segments.append(f'twisting_{twist_scale:02.2f}')

        if not twist_rot:
            name_segments.append(f"noTwistRot")
        if not twist_trans:
            name_segments.append(f"noTwistTrans")

        if num_rots != 1:
            name_segments.append(f"numRots{num_rots:04d}")

        if max_offsets != 1:
            name_segments.append(f"maxOffsets{max_offsets:04d}")

    if sampler._infer_conf.motif_scaffolding.use_replacement:
        name_segments.append('replacement')
        if not sampler._infer_conf.motif_scaffolding.use_contig_for_placement:
            name_segments.append('notContigPlaced')
    else:
        assert sampler._infer_conf.motif_scaffolding.use_twisting, "must use either twisting or replacement method"

    if sampler._infer_conf.motif_scaffolding.use_contig_for_placement:
        name_segments.append('contigPlaced')

    if no_self_cond:
        name_segments.append("noSelfCond")

    if noise_scale != 0.5:
        # fortmat noise scale to have 1 leading digit and 2 decimal places
        name_segments.append(f"noiseScale{noise_scale:02.2f}")

    if min_t != 0.01:
        name_segments.append(f"minT{min_t:02.2f}")

    if num_steps != 200:
        name_segments.append(f"numSteps{num_steps}")

    if N_particles != 1:
        name_segments.append(f"Particles{N_particles:03d}")
        if insert_motif_at_t0:
            name_segments.append(f"insertMotifAtT0")
        if resample_threshold != 0.5:
            name_segments.append(f"resample{resample_threshold:02.2f}")

    if folding_method != 'af2':
        name_segments.append(f"foldingMethod{folding_method}")

    if sampler._infer_conf.motif_scaffolding.keep_motif_seq:
        name_segments.append("keep_motif_seq")

    output_dir = os.path.join(output_dir_stem, '_'.join(name_segments))
    return output_dir


class Sampler:

    def __init__(
            self,
            conf: DictConfig,
            conf_overrides: Dict=None
        ):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
            conf_overrides: Dict of fields to override with new values.
        """
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference
        self._diff_conf = self._infer_conf.diffusion
        self._sample_conf = self._infer_conf.samples

        self._rng = np.random.default_rng(self._infer_conf.seed)

        # Set model hub directory for ESMFold.
        torch.hub.set_dir(self._infer_conf.pt_hub_dir)

        # Set-up accelerator
        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._infer_conf.gpu_id}'
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')

        # Set-up directories
        self._weights_path = self._infer_conf.weights_path

        output_dir =self._infer_conf.output_dir
        if self._infer_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._infer_conf.name
        self._output_dir = os.path.join(output_dir, dt_string)
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')
        self._pmpnn_dir = self._infer_conf.pmpnn_dir

        config_path = os.path.join(self._output_dir, 'inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')

        # Load models and experiment
        self._load_ckpt(conf_overrides)
        self._folding_model = esm.pretrained.esmfold_v1().eval()
        self._folding_model = self._folding_model.to(self.device)

    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        self._log.info(f'Loading weights from {self._weights_path}')

        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(
            self._weights_path, use_torch=True,
            map_location=self.device)

        # Merge base experiment config with checkpoint config.
        self._conf.model = OmegaConf.merge(
            self._conf.model, weights_pkl['conf'].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_se3_diffusion.Experiment(
            conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {
            k.replace('module.', ''):v for k,v in model_weights.items()}
        self.model.load_state_dict(model_weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

    def sample_random_motif_locations(self, motif_segments, length, batch_size):
        segment_lengths = [motif_segment.shape[0] for motif_segment in motif_segments]
        motif_locations_all = []

        # Iterate through all possible permutations of segment_lengths
        for segment_lengths_ in itertools.permutations(segment_lengths):
            motif_locations_order = twisting.get_all_motif_locations(
                length, segment_lengths_, max_offsets=batch_size
            )
            motif_locations_all += motif_locations_order

        # Sample batch_size number of motif locations
        motif_locations = random.sample(motif_locations_all, batch_size)
        return motif_locations


    # Define a helper function that returns the locations of the motif segments (rather than doing it within run_sampling)
    def motif_locations_and_length(self, motif_segments, motif_contig_info, num_motif_locations):
        """motif_locations_and_length samples locations for the motif segments

        Args:
            motif_segments: list of motif segments, each segment is a tuple of (st, end)
            motif_contig_info: dict of motif contig info
            num_motif_locations: number of motif locations to sample
        """
        if self._infer_conf.motif_scaffolding.use_contig_for_placement:
            length = motif_contig_info['total_length']
            motif_locations  = []
            for _ in range(num_motif_locations):
                sample_contig, _, _ = eu.get_sampled_mask(motif_contig_info['contig'], [length, length+1])
                motif_locations.append(save_motif_segments.motif_locations_from_contig(sample_contig[0]))
        else:
            # In this case we keep only the order of motif segments fixed and from the contig.
            length = motif_contig_info['length_fixed']
            motif_locations = self.sample_random_motif_locations(motif_segments, length, num_motif_locations)
        return motif_locations, length


    def run_sampling(self,
        motif_contig_info_row,
        batch_size=1,
        num_backbones=1,
     ):
        """Sets up inference run.

        Args:
            motif_segments: list of tensor7 arrays for motif segments
            motif_locations: list of motif locations [(st_1, end_1),  ..., (st_N, end_N)]
            batch_size: number of backbones to sample in parallel

        All outputs are written to
            {output_dir}/{date_time}
        where {output_dir} is created at initialization.
        """
        # Load in contig separately for each batch in order to sample a
        # different length for the motif in the case that the contig is used
        motif_contig_info = save_motif_segments.load_contig_test_case(motif_contig_info_row)
        length = motif_contig_info['length_fixed'] # this is at odds, with using the variable length contig possibility as used by RFdiffusion
        motif_segments = [torch.tensor(motif_segment, dtype=torch.float64) for motif_segment in motif_contig_info['motif_segments']]
        rigids_motif = torch.cat([motif_segment.to(self.device) for motif_segment in motif_segments], dim=0)
        rigids_motif = eu.remove_com_from_tensor_7(rigids_motif)

        # Confirm that num_backbones is a multiple of the batch_size
        assert num_backbones % batch_size == 0, f"num_backbones {num_backbones} must be a multiple of batch_size {batch_size}"
        num_batches = num_backbones // batch_size
        for batch in range(num_batches):

            # Sample and print motif locations
            if self._infer_conf.motif_scaffolding.use_replacement or \
                self._infer_conf.motif_scaffolding.use_contig_for_placement:
                # If number of particles is greater than 1, we use only one motif location
                if self._infer_conf.particle_filtering.number_of_particles > 1:
                    num_motif_locations = 1
                else:
                    num_motif_locations = batch_size
                motif_locations, length = self.motif_locations_and_length(
                    motif_segments, motif_contig_info, num_motif_locations)
            else:
                motif_locations = None
                length = motif_contig_info['length_fixed']

            if self._infer_conf.motif_scaffolding.use_replacement:
                assert motif_locations is not None, "Must provide motif locations for replacement method"
                motif_mask = torch.zeros([batch_size, length], dtype=torch.bool)
                for b, motif_locations_b in enumerate(motif_locations):
                    for (st, end) in motif_locations_b:
                        motif_mask[b, st:end+1] = True
            else:
                motif_mask = None

            # Make output directories for samples, and continue if they already exist
            length_dir = os.path.join(self._output_dir, f'length_{length}')
            os.makedirs(length_dir, exist_ok=True)
            sample_bidx_id_dir = [
                (b, sample_id, os.path.join(length_dir, f'sample_{sample_id}'))
                for b in range(batch_size) for sample_id in range(batch*batch_size, (batch+1)*batch_size)
            ]
            if all([os.path.isdir(sample_dir) for _, _, sample_dir in sample_bidx_id_dir]):
                self._log.info(f'Skipping sampling for length {length}, batch {batch} because all samples already exist')
                continue

            # Compute vectorized function for degrees of freedom computation
            F, all_motif_locations = twisting.motif_offsets_and_rots_vec_F(
                length, motif_segments, motif_locations=motif_locations, num_rots=self._infer_conf.motif_scaffolding.num_rots,
                device=self.device, max_offsets=self._infer_conf.motif_scaffolding.max_offsets)

            self._log.info(f'Beginning sampling')
            sample_output = self.sample(
                length, rigids_motif=rigids_motif, F=F,
                batch_size=batch_size,
                motif_locations=motif_locations,
                )


            # Save out sample and motif info
            for b, sample_id, sample_dir in sample_bidx_id_dir:
                if os.path.isdir(sample_dir):
                    continue
                os.makedirs(sample_dir, exist_ok=True)

                # Compute / extract motif mask from output
                if motif_mask is None:
                    max_log_p_idx = sample_output['max_log_p_idx'][b]
                    motif_locations_b = all_motif_locations[max_log_p_idx]
                else:
                    assert motif_locations is not None, "Must provide motif locations for replacement method"
                    motif_locations_b = motif_locations[b]

                # Save motif locations to file
                motif_segments_string = ",".join([f"{st}_{end}" for st, end in motif_locations_b])
                motif_segments_fn = os.path.join(sample_dir, f"motif_segments_{sample_id}.txt")
                with open(motif_segments_fn, "w") as f:
                    f.write(motif_segments_string)

                # Set positions of the motif in the sequence to the prescribed amino acids
                aatype = np.ones([length], dtype=np.int32)
                for i, (st, end) in enumerate(motif_locations_b):
                    aatype[st:end+1] = motif_contig_info['aatype'][i]

                if self._infer_conf.aux_traj:
                    au.save_traj(
                        sample_output['prot_traj'][:, b],
                        sample_output['rigid_0_traj'][:, b],
                        aatype=aatype,
                        diffuse_mask=np.ones(length),
                        output_dir=sample_dir,
                    )
                else:
                    # Use b-factors to specify which residues are diffused.
                    b_factors = np.tile((np.ones(length) * 100)[:, None], (1, 37))
                    sample_path = os.path.join(sample_dir, 'sample')
                    au.write_prot_to_pdb(
                        sample_output['prot_traj'][:, b][0],
                        sample_path,
                        aatype=aatype,
                        b_factors=b_factors
                    )
            self._log.info(f'Done sampling batch {batch} of {num_batches}')

        self._log.info(f'Beginning validation')
        _ = self.run_self_consistency(
            output_dir=self._output_dir,
            rigids_motif=rigids_motif,
            use_motif_seq=self._infer_conf.motif_scaffolding.keep_motif_seq,
            motif_contig_info=motif_contig_info,
        )

    def replacement_sample(self, sample_feats, t):
        batch_size = sample_feats['rigids_t'].shape[0]
        for b in range(batch_size):
            motif_noised = self.exp.diffuser.forward_marginal(
                ru.Rigid.from_tensor_7(sample_feats['rigids_motif']),
                t, as_tensor_7=True)['rigids_t']

            # Create a list of indices of the motif locations by concatenating the ranges in the motif_locations,
            # and use this to set corresponding indices in rigids_t to the noised motif
            motif_idcs = [range(st, end+1) for st, end in sample_feats['motif_locations'][b]]
            motif_idcs = [item for sublist in motif_idcs for item in sublist]

            sample_feats['rigids_t'][b, motif_idcs, :] = motif_noised.to(
                sample_feats['rigids_t'].device
            ).to(sample_feats['rigids_t'].dtype)
        rigids_t = ru.Rigid.from_tensor_7(sample_feats['rigids_t'])
        sample_feats['R_t'] = rigids_t.get_rots().get_rot_mats().to(torch.float64)
        sample_feats['trans_t'] = rigids_t.get_trans().to(torch.float64)
        return sample_feats

    def run_self_consistency(
            self,
            output_dir: str,
            rigids_motif: torch.tensor,
            use_motif_seq=True,
            motif_contig_info=None,
            ):
        """Run self-consistency on design proteins against reference protein.

        Args:
            output_dir: directory with subdirectory for each sampled length (possibly just one) each containing backbone samples
            reference_pdb_path: path to reference protein file
            motif_mask: Optional mask of which residues are the motif.
            rigids_motif: ground truth motif coordinates
            use_motif_seq: use the motif sequence as the fixed sequence rather than redesigning
            motif_sequence: sequence of integers amino acids in the motif,
                list of np array arrays for each motif segment

        Returns:
            Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
            Writes ESMFold outputs to decoy_pdb_dir/esmf (or AF2 outputs to decoy_pdb_dir/af2)
            Writes results in decoy_pdb_dir/sc_results.csv
        """
        # Run Protein MPNN on each pdb
        decoy_pdb_dirs = []
        for decoy_pdb_dir in eu.list_subdirectories(output_dir):
            # add decoy pdb dir to list if it contains a pdb file
            pdb_path = os.path.join(decoy_pdb_dir, "sample_1.pdb")
            if not os.path.exists(pdb_path):
                continue

            # Check that a file within decoy_pdb_dir contains the motif segments
            motif_segments_fns = [fn for fn in os.listdir(decoy_pdb_dir) if "motif_segments" in fn]
            if len(motif_segments_fns) == 0:
                self._log.info(f"No motif segments found in {decoy_pdb_dir}.  Skipping.")
                continue


            ## Check in sc_results.csv exists in all (or any) of the decoy_pdb_dirs.
            # In some directories it will exist but have only a header row.  In this case we check that the numer of lines > 2, and delete the file if so.
            if os.path.exists(os.path.join(decoy_pdb_dir, "sc_results.csv")):
                with open(os.path.join(decoy_pdb_dir, "sc_results.csv"), 'r') as f:
                    lines = f.readlines()
                if len(lines) > 2:
                    self._log.info(f"Self-consistency already run on {decoy_pdb_dir}.  Skipping.")
                    continue
                else:
                    os.remove(os.path.join(decoy_pdb_dir, "sc_results.csv"))

            decoy_pdb_dirs.append(decoy_pdb_dir)

        if all([os.path.exists(os.path.join(d, "sc_results.csv")) for d in decoy_pdb_dirs]):
            self._log.info(f"Self-consistency already run on all decoy pdb dirs.  Skipping.")
            return

        # Run Protein MPNN
        for decoy_pdb_dir in decoy_pdb_dirs:
            sc_output_dir = os.path.join(decoy_pdb_dir, 'self_consistency')
            os.makedirs(sc_output_dir, exist_ok=True)

            # Copy pdb into sc dir
            pdb_path = os.path.join(decoy_pdb_dir, "sample_1.pdb")

            shutil.copy(pdb_path, os.path.join(sc_output_dir, os.path.basename(pdb_path)))
            self.run_MPNN(
                decoy_pdb_dir,
                use_motif_seq=use_motif_seq,
                motif_contig_info=motif_contig_info,
                )

        # Run Folding (EMSFold -- to add AF2)
        if self._infer_conf.folding_method == "esmf":
            self._log.info(f'Running esmf')
            structure_pred_dir = "esmf/"
            for decoy_pdb_dir in decoy_pdb_dirs:
                # Run ESM on all MPNN seqences in decoy_pdb_dir
                self.run_ESMfold(decoy_pdb_dir)
        else:
            assert self._infer_conf.folding_method == "af2"
            structure_pred_dir = "af2/"
            self._log.info(f'Running AF2')
            print("Add or find base dir in inference config!")
            af2_script_path = "./run_af2_on_all_samples.sh"
            cmd =  ['bash', af2_script_path, output_dir]
            print("af2 cmd: ", af2_script_path)
            ret = -1
            while ret < 0:
                try:
                    process = subprocess.Popen(cmd)
                    ret = process.wait()
                except Exception as e:
                    print("caught exception!!", e)
                    traceback.print_exc()

                    num_tries += 1
                    self._log.info(f'AF2 failed. Attempt {num_tries}/5')
                    torch.cuda.empty_cache()
                    if num_tries > 4:
                        raise e

        # Compute metrics and compile self-consistency results
        for decoy_pdb_dir in decoy_pdb_dirs:
            self.compute_and_compile_self_consistency(
                decoy_pdb_dir, rigids_motif, structure_pred_dir=structure_pred_dir)

    def compute_and_compile_self_consistency(self, decoy_pdb_dir, rigids_motif, structure_pred_dir="esmf/"):
        reference_pdb_path = os.path.join(decoy_pdb_dir, "sample_1.pdb")
        sample_feats = du.parse_pdb_feats('sample', reference_pdb_path)
        structure_pred_dir_full = os.path.join(decoy_pdb_dir, "self_consistency", structure_pred_dir)

        mpnn_results = {
            'tm_score': [],
            'sample_path': [],
            'rmsd': [],
            'motif_rmsd': []
        }
        for struct_pred_path in os.listdir(structure_pred_dir_full):
            # Skip saved motifs
            if "motif" in struct_pred_path or "sample" not in struct_pred_path:
                continue
            seq_id = struct_pred_path.strip(".pdb").split("_")[-1]
            struct_pred_path_full = os.path.join(structure_pred_dir_full, struct_pred_path)
            esmf_feats = du.parse_pdb_feats('folded_sample', struct_pred_path_full)
            sample_seq = du.aatype_to_seq(sample_feats['aatype'])

            # Calculate scTM of ESMFold outputs with reference protein
            _, tm_score = metrics.calc_tm_score(
                sample_feats['bb_positions'], esmf_feats['bb_positions'],
                sample_seq, sample_seq)
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats['atom_positions'][..., :3, :].reshape([-1, 3]),
                    esmf_feats['atom_positions'][..., :3, :].reshape([-1, 3])
                )
            motif_segments_fn = [fn for fn in os.listdir(decoy_pdb_dir) if "motif_segments" in fn][0]
            motif_segments_fn = os.path.join(decoy_pdb_dir, motif_segments_fn)
            motif_locations = eu.load_motif_locations(motif_segments_fn)

            of_motif_all_atom = np.concatenate([
                esmf_feats['atom_positions'][st:end+1] for st, end in motif_locations]
            )
            au.write_prot_to_pdb(
                of_motif_all_atom,
                    decoy_pdb_dir + f"/self_consistency/{structure_pred_dir}/sampled_motif_{seq_id}.pdb",
                    no_indexing=True,
                    overwrite=True
                )
            of_motif_bb_positions = of_motif_all_atom[:, :3, :].reshape([-1, 3])
            assert len(of_motif_bb_positions.shape) == 2

            # Save pdb file with motif segments concatenated together
            psis = torch.zeros_like(rigids_motif[:, :2])
            atom37_motif = all_atom.compute_backbone(
                ru.Rigid.from_tensor_7(rigids_motif), psis)[0]
            true_motif_bb_positions = atom37_motif[:, :3].reshape([-1, 3])

            # Concatenate motif segments
            motif_rmsd = metrics.calc_aligned_rmsd(
                true_motif_bb_positions.numpy(), of_motif_bb_positions)
            mpnn_results['motif_rmsd'].append(motif_rmsd)
            mpnn_results['rmsd'].append(rmsd)
            mpnn_results['tm_score'].append(tm_score)
            mpnn_results['sample_path'].append(struct_pred_path_full)

        # Save results to CSV
        csv_path = os.path.join(decoy_pdb_dir, 'sc_results.csv')
        print("mpnn_results:", mpnn_results)
        mpnn_results = pd.DataFrame(mpnn_results)
        mpnn_results.to_csv(csv_path)


    def run_ESMfold(self, decoy_pdb_dir):
        mpnn_fasta_path = os.path.join(decoy_pdb_dir, "self_consistency/seqs/sample_1.fa")
        esmf_dir = os.path.join(decoy_pdb_dir, 'self_consistency/esmf')
        os.makedirs(esmf_dir, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        esmf_sample_paths = []
        for i, (_, string) in enumerate(fasta_seqs.items()):
            esmf_sample_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
            if not os.path.exists(esmf_sample_path):
                _ = self.run_folding(string, esmf_sample_path)
            esmf_sample_paths.append(esmf_sample_path)
        return esmf_sample_paths


    def run_MPNN(
            self,
            decoy_pdb_dir: str,
            use_motif_seq=True,
            fixed_positions_list=None,
            motif_contig_info=None,
            ):
        """Run self-consistency on design proteins against reference protein.

        Args:
            decoy_pdb_dir: directory where designed protein files are stored.
            use_motif_seq: use the motif sequence as the fixed sequence rather than redesigning
            fixed_positions_list: list of indices of positions to
                keep fixed in the sequence when MPNN redesigning.
        """
        # check if there are indices to redesign
        motif_idcs_to_redesign = self._infer_conf.motif_scaffolding.keep_motif_seq
        # check that motif_contig_info['idcs_to_redesign'] is a string
        if motif_idcs_to_redesign and isinstance(motif_contig_info['idcs_to_redesign'], str):
            motif_idcs_to_redesign = motif_idcs_to_redesign and (len(motif_contig_info['idcs_to_redesign'])>1)
        else:
            motif_idcs_to_redesign = False

        if motif_idcs_to_redesign:
            motif_segments_fn = [fn for fn in os.listdir(decoy_pdb_dir) if "motif_segments" in fn][0]
            motif_segments_fn = os.path.join(decoy_pdb_dir, motif_segments_fn)
            motif_locations = eu.load_motif_locations(motif_segments_fn)
            fixed_positions_list = mu.seq_indices_to_fix(
                motif_locations, motif_contig_info['contig'],
                motif_contig_info['idcs_to_redesign'])
        else:
            fixed_positions_list = None

        sc_dir = os.path.join(decoy_pdb_dir, "self_consistency")
        # Run ProteinMPNN
        output_path = os.path.join(sc_dir, "parsed_pdbs.jsonl")
        reference_pdb_path = os.path.join(decoy_pdb_dir, "sample_1.pdb")
        if not os.path.exists(output_path):
            process = subprocess.Popen([
                'python',
                f'{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
                f'--input_path={decoy_pdb_dir}',
                f'--output_path={output_path}',
            ])
            _ = process.wait()
            num_tries = 0

        # Use known fixed motif sequence
        if use_motif_seq and fixed_positions_list is not None:
            path_for_assigned_chains = os.path.join(sc_dir, "parsed_pdbs_assigned_chains.jsonl")
            process = subprocess.Popen([
                'python',
                f'{self._pmpnn_dir}/helper_scripts/assign_fixed_chains.py',
                f'--input_path={output_path}',
                f'--output_path={path_for_assigned_chains}',
                f'--chain_list',
                "A"])

            path_for_fixed_positions = os.path.join(sc_dir, "fixed_positions.jsonl")
            process = subprocess.Popen([
                'python',
                f'{self._pmpnn_dir}/helper_scripts/make_fixed_positions_dict.py',
                f'--input_path={output_path}',
                f'--output_path={path_for_fixed_positions}',
                f'--chain_list',
                "A",
                f'--position_list',
                " ".join([str(char) for char in fixed_positions_list])
            ])


        ret = -1
        pmpnn_args = [
            'python',
            f'{self._pmpnn_dir}/protein_mpnn_run.py',
            '--out_folder',
            sc_dir,
            '--jsonl_path',
            output_path,
            '--num_seq_per_target',
            str(self._sample_conf.seq_per_sample),
            '--sampling_temp',
            '0.1',
            '--seed',
            '38',
            '--batch_size',
            '1',
        ]
        if use_motif_seq and fixed_positions_list is not None:
            pmpnn_args += [
                '--fixed_positions',
                path_for_fixed_positions,
                '--chain_id_jsonl',
                path_for_assigned_chains,
            ]
        if "cuda" in self.device:
            pmpnn_args.append('--device')
            pmpnn_args.append(self.device.split(":")[1])

        while ret < 0:
            print("running ProteinMPNN", pmpnn_args)
            try:
                process = subprocess.Popen(
                    pmpnn_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                ret = process.wait()
            except Exception as e:
                print("caught exception!!", e)
                traceback.print_exc()

                num_tries += 1
                self._log.info(f'Failed ProteinMPNN. Attempt {num_tries}/5')
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e
        mpnn_fasta_path = os.path.join(
            sc_dir,
            'seqs',
            os.path.basename(reference_pdb_path).replace('.pdb', '.fa')
        )
        return mpnn_fasta_path



    def run_folding(self, sequence, save_path):
        """Run ESMFold on sequence."""
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w") as f:
            f.write(output)
        return output

    def shuffle_feats_from_resample(self, resample_indices):
        """shuffle_feats_from_resample shuffles sample feats and model feats according to resample_idcs
        """
        P = len(resample_indices)
        if 'model_out' in self.PF_cache and "twist_log_p" in self.PF_cache['model_out']:
            model_out_prev = self.PF_cache['model_out']
            for k, v in model_out_prev.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == P:
                    model_out_prev[k] = v[resample_indices]

        sample_feats = self.PF_cache['sample_feats']
        for k, v in sample_feats.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == P:
                if k == "rigids_motif": continue
                sample_feats[k] = v[resample_indices]

    def M_T(self, P):
        """M_T is the Feynman-Kac initial distribution
        """
        sample_length = self.PF_cache['length']
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length*P,
            as_tensor_7=True)
        rigid_t = Rigid.from_tensor_7(ref_sample['rigids_t'].reshape(P, sample_length, 7))
        self.PF_cache['sample_feats']['R_t'] = rigid_t.get_rots().get_rot_mats().to(torch.float64)
        self.PF_cache['sample_feats']['trans_t'] = rigid_t.get_trans().to(torch.float64)
        self.PF_cache['sample_feats']['rigids_t'] = rigid_t.to_tensor_7().to(torch.float64)
        sample_feats = self.PF_cache['sample_feats']

        # Move all tensors in sample_feats onto device
        for k, v in sample_feats.items():
            if isinstance(v, torch.Tensor):
                sample_feats[k] = v.to(self.device)

        t_cts = 1.0
        self_conditioning = self.exp._model_conf.embed.embed_self_conditioning and not self._infer_conf.motif_scaffolding.no_self_conditioning
        if self_conditioning:
            sample_feats = self.exp._set_t_feats(
                sample_feats, t_cts, sample_feats['t_placeholder'])
            sample_feats = self.exp._self_conditioning(sample_feats)
            self.PF_cache['sample_feats'] = sample_feats
        return rigid_t.to_tensor_7().cpu().detach(), {}


    def M(
        self,
        t,
        xtp1,
        extra_vals=None,
        P=None
    ):
        """M is the Feynman-Kac transition kernel.

        Args:
            t: integer time step
            xtp1: rigids defining backbone poses at time t+1 of shape [P, L, 7]
            extra_vals: dictionary of extra values passed to the transition kernel
            P: number of particles
        """
        if t == self._diff_conf.num_t: return self.M_T(P)

        # If a resampling step was performed in the previous time step, then sample all features in sample_feats accordingly
        # Move over previous twisting log potential if present
        resample_indices = extra_vals[('resample_idcs', t+1)]
        self.shuffle_feats_from_resample(resample_indices)

        min_t = self._diff_conf.min_t
        t_cts = np.linspace(min_t, 1.0, self._diff_conf.num_t+1)[t]
        dt = 1/self._diff_conf.num_t

        model_out = self.PF_cache['model_out']
        sample_feats = self.PF_cache['sample_feats']
        rigid_pred = model_out['rigids']

        diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
        rigids_t, log_Mt = self.exp.diffuser.reverse(
            rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
            rot_score=model_out['rot_score'],
            trans_score=model_out['trans_score'],
            diffuse_mask=diffuse_mask,
            t=t_cts,
            dt=dt,
            noise_scale=self._diff_conf.noise_scale,
            return_log_p_sample=True,
        )
        self.PF_cache['log_Mt'] = log_Mt

        if self._infer_conf.aux_traj:
            # Calculate x0 prediction derived from score predictions.
            if not "aux_traj" in self.PF_cache:
                self.PF_cache['aux_traj'] = {'all_bb_0_pred': [], 'all_bb_prots': []}
                self.PF_cache['max_log_p_idx_by_t'] = []
            self.PF_cache['aux_traj']['all_bb_0_pred'].append(du.move_to_np(all_atom.compute_backbone(
                ru.Rigid.from_tensor_7(rigid_pred), model_out['psi_pred'])[0]))
            self.PF_cache['aux_traj']['all_bb_prots'].append(du.move_to_np(all_atom.compute_backbone(
                rigids_t, model_out['psi_pred'])[0]))
            self.PF_cache['max_log_p_idx_by_t'].append(model_out['max_log_p_idx'])


        # If the last step, return the model output not noised rigids
        if t==0:
            rigids_t = rigid_pred.cpu().detach()
        else:
            rigids_t = rigids_t.to_tensor_7().cpu().detach()

        # Add prot_traj and psi_pred with appropriate dimensios 0for validation
        self.PF_cache['prot_traj'] = du.move_to_np(
            all_atom.compute_backbone(ru.Rigid.from_tensor_7(
                rigids_t), model_out['psi_pred'])[0])[None]
        model_out['psi_pred'] = model_out['psi_pred'][None]
        return rigids_t.detach(), {}

    def G(
        self,
        t,
        xtp1,
        xt,
        extra_vals=None,
    ):
        """G is the log potential function.
        """
        num_t = self._diff_conf.num_t
        min_t = self._diff_conf.min_t
        t_cts = np.linspace(min_t, 1.0, num_t+1)[t]
        dt = 1/num_t

        # Extract and update sample feats
        sample_feats = self.PF_cache['sample_feats']
        for k, v in sample_feats.items():
            if isinstance(v, torch.Tensor): sample_feats[k] = v.detach()

        # Update sample feats with new rigids and t feature
        sample_feats['rigids_t'] = xt.to(self.device)
        xt_rigids = Rigid.from_tensor_7(xt)
        sample_feats['R_t'] = xt_rigids.get_rots().get_rot_mats().to(self.device).to(torch.float64)
        sample_feats['trans_t'] = xt_rigids.get_trans().to(self.device).to(torch.float64)
        sample_feats = self.exp._set_t_feats(sample_feats, t_cts, sample_feats['t_placeholder'])
        self.PF_cache['sample_feats'] = sample_feats

        # Run model
        model_out = self.exp.model(
            sample_feats, F=self.PF_cache['F'],
            use_twisting=self._infer_conf.motif_scaffolding.use_twisting,
            twist_scale=self._infer_conf.motif_scaffolding.twist_scale,
            twist_potential_rot=self._infer_conf.motif_scaffolding.twist_potential_rot,
            twist_potential_trans=self._infer_conf.motif_scaffolding.twist_potential_trans,
            twist_update_rot=self._infer_conf.motif_scaffolding.twist_update_rot,
            twist_update_trans=self._infer_conf.motif_scaffolding.twist_update_trans,
        )
        self_conditioning = self.exp._model_conf.embed.embed_self_conditioning and not self._infer_conf.motif_scaffolding.no_self_conditioning
        if self_conditioning: sample_feats['sc_ca_t'] = model_out['rigids'][..., 4:]

        # Detach model outputs and add to cache
        for k, v in model_out.items():
            if isinstance(v, torch.Tensor): model_out[k] = v.detach()
        self.PF_cache['model_out'] = model_out

        log_G = self.PF_cache['model_out']['twist_log_p'].cpu()
        if t != self._diff_conf.num_t:
            log_G = log_G - self.PF_cache['twist_log_p_tp1'].cpu()
            log_p_model_uncond = self.exp.diffuser.reverse_log_prob(
                rigid_t=xtp1,
                rigid_tm1=xt,
                rot_score=self.PF_cache['model_out']['rot_score_uncond'],
                trans_score=self.PF_cache['model_out']['trans_score_uncond'],
                t=t_cts,
                dt=dt,
                )
            log_G_proposal = log_p_model_uncond.cpu() - self.PF_cache['log_Mt'].cpu()
            assert log_G_proposal.shape[0] == log_G.shape[0]
            assert log_G_proposal.shape[0] == xtp1.shape[0]
            log_G = log_G + log_G_proposal
        self.PF_cache['twist_log_p_tp1'] = self.PF_cache['model_out']['twist_log_p'].detach().cpu()
        return log_G.detach()

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=True,
            aux_traj=False,
            noise_scale=1.0,
            F=None,
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)

        device = sample_feats['rigids_t'].device
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            # batched
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        if num_t is None:
            num_t = self._diff_conf.num_t
        if min_t is None:
            min_t = self._diff_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []

        self_conditioning = self.exp._model_conf.embed.embed_self_conditioning and not self._infer_conf.motif_scaffolding.no_self_conditioning
        if self_conditioning:
            sample_feats = self.exp._set_t_feats(
                sample_feats, reverse_steps[0], t_placeholder)
            sample_feats = self.exp._self_conditioning(sample_feats)
        #prev_t = 1.
        for t in reverse_steps:
            # Detach all tensors in sample_feats.
            for k, v in sample_feats.items():
                if isinstance(v, torch.Tensor): sample_feats[k] = v.detach()

            # if the conditional_method is 'replacement', sample the motif part from the forward diffusion
            if self._infer_conf.motif_scaffolding.use_replacement:
                sample_feats = self.replacement_sample(sample_feats, t)

            sample_feats = self.exp._set_t_feats(sample_feats, t, t_placeholder)

            model_out = self.exp.model(
                sample_feats, F=F,
                use_twisting=self._infer_conf.motif_scaffolding.use_twisting,
                twist_scale=self._infer_conf.motif_scaffolding.twist_scale,
                twist_potential_rot=self._infer_conf.motif_scaffolding.twist_potential_rot,
                twist_potential_trans=self._infer_conf.motif_scaffolding.twist_potential_trans,
                twist_update_rot=self._infer_conf.motif_scaffolding.twist_update_rot,
                twist_update_trans=self._infer_conf.motif_scaffolding.twist_update_trans,
            )
            rigid_pred = model_out['rigids']
            if t > min_t:
                if self_conditioning:
                    sample_feats['sc_ca_t'] = rigid_pred[..., 4:]
                fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                rigids_t = self.exp.diffuser.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                    rot_score=model_out['rot_score'],
                    trans_score=model_out['trans_score'],
                    diffuse_mask=diffuse_mask,
                    t=t,
                    dt=dt,
                    center=center,
                    noise_scale=noise_scale,
                )
            else:
                rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])

                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                psi_pred = model_out['psi_pred']

            sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
            sample_feats['R_t'] = rigids_t.get_rots().get_rot_mats().to(device).to(torch.float64)
            sample_feats['trans_t'] = rigids_t.get_trans().to(device).to(torch.float64)

            if aux_traj:
                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                psi_pred = model_out['psi_pred']

                all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))
                atom37_0 = all_atom.compute_backbone(
                    ru.Rigid.from_tensor_7(rigid_pred),
                    psi_pred
                )[0]
                all_bb_0_pred.append(du.move_to_np(atom37_0))
                all_trans_0_pred.append(du.move_to_np(trans_pred_0))

                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))


        if not aux_traj:
            atom37_t = all_atom.compute_backbone(
                rigids_t, psi_pred)[0]
            all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            'prot_traj': all_bb_prots,
        }
        if not self._infer_conf.motif_scaffolding.use_replacement:
            ret['max_log_p_idx'] = model_out['max_log_p_idx']

        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['psi_pred'] = psi_pred[None]
            ret['rigid_0_traj'] = all_bb_0_pred

        return ret



    def sample(self, sample_length: int, rigids_motif: torch.Tensor = None, F: torch.Tensor = None,
                    batch_size: int = 1, motif_locations: list = None
    ):
        """Sample based on length.

        Args:
            sample_length: length to sample

        Returns:
            Sample outputs.
        """
        # Process motif features.
        res_mask = np.ones([batch_size, sample_length])
        fixed_mask = np.zeros_like(res_mask)

        init_feats = {
            'res_mask': res_mask,
            'seq_idx': torch.arange(1, sample_length+1).repeat(batch_size, 1),
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': np.zeros((batch_size, sample_length, 7, 2)),
            'sc_ca_t': np.zeros((batch_size, sample_length, 3)),
            'motif_locations': motif_locations, # list of length batch_size of lists of motif segment locations
        }

        # Add rotation and translation features
        # sample a vector of shape (batch_size * sample_length, 7) and then reshape.
        ref_sample = self.diffuser.sample_ref(
            n_samples=sample_length*batch_size,
            as_tensor_7=True,)
        rigid_t = Rigid.from_tensor_7(ref_sample['rigids_t'].reshape(batch_size, sample_length, 7))
        init_feats['R_t'] = rigid_t.get_rots().get_rot_mats().to(torch.float64)
        init_feats['trans_t'] = rigid_t.get_trans().to(torch.float64)
        init_feats['rigids_t'] = rigid_t.to_tensor_7().to(torch.float64)

        # Add move to torch and GPU
        init_feats = tree.map_structure(lambda x: x if (x is None or torch.is_tensor(x)) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(lambda x: x if x is None else x.to(self.device), init_feats)

        # Add motif into init_feats (no batch dimension)
        if rigids_motif is not None: init_feats['rigids_motif'] = rigids_motif

        # Run inference
        sample_out = self.inference_fn(
            init_feats,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t,
            aux_traj=self._infer_conf.aux_traj,
            noise_scale=self._diff_conf.noise_scale,
            F=F,
        )

        # Remove batch dimension ? (not sure why this is here)
        return sample_out

@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Sampler(conf)
    torch.set_default_tensor_type('torch.FloatTensor')

    output_dir_stem = sampler._output_dir


    # Load motif test case details
    inpaint_df = pd.read_csv(sampler._infer_conf.motif_scaffolding.inpaint_cases_csv)
    contigs_by_test_case = save_motif_segments.load_contigs_by_test_case(inpaint_df)
    if sampler._infer_conf.motif_scaffolding.test_name is not None:
        test_names = [sampler._infer_conf.motif_scaffolding.test_name]
    else:
        test_names = ["2KL8", "1BCF", "1PRW", "6EXZ_long"]# , "6EXZ_short", "1YCR", "5TPN", "7MRX_85"]
        test_names = ["1PRW", "1QJG", "5TRV_short"]# , "6EXZ_short", "1YCR", "5TPN", "7MRX_85"]
        test_names = [name for name in contigs_by_test_case.keys()]

    for test_name in test_names:
        print("starting test case: ", test_name)

        motif_contig_info_row = list(inpaint_df[inpaint_df.target==test_name].iterrows())[0][1]
        sampler._output_dir = construct_output_dir(
            sampler, test_name, output_dir_stem)
        os.makedirs(sampler._output_dir, exist_ok=True)

        sampler.run_sampling(
            motif_contig_info_row=motif_contig_info_row,
            batch_size=sampler._infer_conf.motif_scaffolding.batch_size,
            num_backbones=sampler._infer_conf.motif_scaffolding.number_of_samples,
            )
        elapsed_time = time.time() - start_time
        print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
