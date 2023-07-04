# Image experiments 

## Code structure

- `image_diffusion/`:  utilities for image diffusion models and SMC samplers. Core files include
    -  `smc_diffusion.py`: base SMC diffusion sampler
    - `feynman_kac_image_ddpm.py`: a series of SMC samplers including TDS, IS, SMC-Diff, Replacement and Unconditional model. 
    - `operators.py`: specifies likelihood constraints and twisting functions 

- `utils/`: utilities for file reading and writing, image writing

- `image_confs/`: configs for model, task and sampler 

- `datasets/`: image datasets

- `models/`: diffusion models and evaluation models 

- `run_image.py`: the main inference file 


## Environment installation
Use [miniconda](https://docs.conda.io/en/main/miniconda.html) to install dependencies:
```bash
conda env create -f image_tds.yml
```

## Pretrained models download

Download the model [here](https://drive.google.com/file/d/18I0v7V_k4p0hY_J0FF1G6X8jVNekyS86/view?usp=sharing) and put them under the `models/` directory.  


## Example commands

To run image class conditional generation experiment:
```
python run_image.py --model_and_diffusion_config image_confs/mnist_model_and_diffusion_conf.yml --task_config image_confs/task_class_cond_gen_conf.yml --sampler_config image_confs/base_sampler_conf.yml 
```

To run image inpainting experiment:
```
python run_image.py --model_and_diffusion_config image_confs/mnist_model_and_diffusion_conf.yml --task_config image_confs/task_inpainting_conf.yml --sampler_config image_confs/base_sampler_conf.yml 
```

