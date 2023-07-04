# Practical and Asymptotically Exact Conditional Sampling in Diffusion Models ([arxiv link](https://arxiv.org/abs/2306.17775))

<img src="https://github.com/blt2114/twisted_diffusion_sampler/blob/main/media/TDS_protein.gif" alt= “” width="50%" height="50%">

The above shows the output of the twisted diffusion sampler (TDS) applied to the motif-scaffolding problem, a central problem in computational protein design. TDS uses a diffusion model of protein structures to evolves a set of a weighted set of scaffolds (red-white-blue) to stabilize a helical motif highlighted in black.  Over the course of the denoising process, promising scaffolds are up-weighted and replicated while less promising scaffolds are down-weighted, fade away, and are are replaced.  Ultimately TDS produces a weighted set of compatible scaffolds.

TDS enables one use existing classification models and diffusion generative models for a range of conditional generation problems without any further neural network training or fine-tuning.  The algorithm applies when one has (1) a diffusion model capable of unconditional generation and (2) a way to specify the conditioning criteria as either a likelihood given (noise-free) data or more general constraints including inpainting. 

The distinguishing feature of TDS is that with larger computational bugets (increasing the numbers of particles) it provides arbitrarily accurate estimates of exact conditional distributions that defined only through the unconditional diffusion model and the likelihood.  This characteristic means that one can trade off computational cost and quality of outputs.  And, when enough compute is used, the outputs of TDS do no depend on potentially subjective or arbitrary hyperparamters of procedure.

To learn more about TDS, read our preprint: "Practical and Asymptotically Exact Conditional Sampling in Diffusion Models" [arxiv link](https://arxiv.org/abs/2306.17775)

## Code structure 

- `smc_utils/`: core utility of SMC algorithms
    - feynman_kac_pf.py
    - smc_utils.py

- `image_exp/`: image experiments

- `protein_exp/`: protein design motif-scaffolding experiments 

See each experiment sub-directory for environment setup and examples. 

## Notes
Please feel welcome to make pull requests for new contributions or bug fixes, or to post issues with questions.

If you use our work then please cite
```
@article{wu2023practical,
  title={Practical and Asymptotically Exact Conditional Sampling in Diffusion Models},
  author={Wu, Luhuan and Trippe, Brian L and Naesseth, Christian A. and Blei, David M and Cunningham, John P},
  journal={arXiv preprint arXiv:2306.17775},
  year={2023}
}

