# Twisted Diffusion Sampling for Motif-Scaffolding



## Code provenance

This repo closely adapted from [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion), which incorporates code from various additional sources including 
[OpenFold](https://github.com/aqlaboratory/openfold),
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN),
[AlphaFold](https://github.com/deepmind/alphafold),
and [geomstats](https://github.com/geomstats/geomstats).
We direct the interested user to [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion) for further information.

## Installation and Inference

### Environment installation

Use [miniconda](https://docs.conda.io/en/main/miniconda.html) to install dependencies:
```bash
conda env create -f se3_tds.yml
```

Next, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```

We also require numba.

```
pip install numba
```
### Inference Command Example
To run TDS for motif scaffolding, you may run `inference_particle_filter.py`.
We have added several [Hydra](https://hydra.cc) configuration options related to TDS;
all default config options and defaults are in `config/inference.yaml`.
As an example, you can try the following run-command.

```
python experiments/inference_particle_filter.py inference.motif_scaffolding.test_name=1QJG 
```

Samples will be saved to `output_dir` in the `inference.yaml`. By default it is
set to `./inference_outputs/`. Sample outputs will be saved as follows,

```shell
inference_outputs
└── run_name                                    # Date time of inference.
    ├── inference_conf.yaml                     # Config used during inference.
    └── motif_name/batch_name/length/           # Nested directories with motif name, batch info and scaffold length
        ├── sample_0                            # Sample ID
        │   ├── sample_1.pdb                    # Final sample
        │   ├── sc_results.csv                  # Summary metrics CSV, including self-consistency and motif-RMSDs 
        │   ├── self_consistency                # Self consistency results        
        │   │   ├── esmf                        # ESMFold predictions using ProteinMPNN sequences
        │   │   │   ├── sample_0.pdb
        │   │   │   ├── sample_1.pdb
        │   │   │   ...
        │   │   ├── parsed_pdbs.jsonl           # Parsed chains for ProteinMPNN
        │   │   ├── sample_1.pdb
        │   │   └── seqs                        
        │   │       └── sample_1.fa             # ProteinMPNN sequences
        ├── sample_1                            # Next samples
        └── ...
```
