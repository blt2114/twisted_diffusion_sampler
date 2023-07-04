import os
import argparse
import yaml 
import random 
import torch as th
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np 
import time
from functools import partial
import sys 

sys.path.append("..") 


from utils import yamlread
from image_diffusion import dist_util
from smc_utils.feynman_kac_pf import smc_FK 
from smc_utils.smc_utils import compute_ess_from_log_w
from image_diffusion.operators import get_operator, ConditioningMethod
from image_diffusion.image_util import get_dataloader, gen_mask, toU8, imwrite 
from image_diffusion.eval_util import pred 

import matplotlib.pyplot as plt 
import fire 


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from image_diffusion.script_util import (
    NUM_CLASSES,
    classifier_defaults,
    create_classifier,
    select_args,
    create_model_and_diffusion, 
    model_and_diffusion_defaults, 
    sampler_defaults, 
    load_classifier, 
)  # noqa: E402


from collections import defaultdict 


class NoneDict(defaultdict):
    def __init__(self, **kwargs):
        super().__init__(self.return_None)
        self.update(**kwargs)

    @staticmethod
    def return_None():
        return None

    def __getattr__(self, attr):
        return self.get(attr)


# utility to update configs parsed from command lines 
def process_configs(model_and_diffusion_config, task_config, sampler_config, 
                    sampler=None, P=None, T=None, var_type=None, tausq=None, ess_threshold=None, 
                    t_truncate_percent=None, mask=None, classifier_guidance_scale=None):

    model_and_diffusion_config_defaults = model_and_diffusion_defaults()
    model_and_diffusion_config_defaults.update(yamlread(model_and_diffusion_config))
    model_and_diffusion_config = model_and_diffusion_config_defaults 

    task_config = yamlread(task_config)

    sampler_config_defaults = sampler_defaults() 
    sampler_config_defaults.update(yamlread(sampler_config))
    sampler_config = sampler_config_defaults

    if sampler is not None:
        sampler_config.update({"name": sampler})
    if T is not None:
        model_and_diffusion_config.update({"timestep_respacing": T})
    if P is not None:
        sampler_config.update(num_particles=P)
    if ess_threshold is not None:
        sampler_config.update(ess_threshold=ess_threshold)
    if t_truncate_percent is not None:
        sampler_config.update(t_truncate_percent=t_truncate_percent)
    
    if sampler_config['ess_threshold'] is None:
        sampler_config.update(ess_threshold=1.0)

    if mask is not None:
        task_config.update(mask=mask)

    # can be set to lower value to reduce memory cost 
    sampler_config['batch_p'] = sampler_config['num_particles']

    # for class conditional generation task 
    if classifier_guidance_scale is not None:
        task_config['classifier_guidance_scale'] = classifier_guidance_scale 


    if var_type is not None:
        task_config.update(pred_xstart_var_type=float(var_type))
    if tausq is not None:
        task_config.update(tausq=float(tausq))

    model_and_diffusion_config.update({"sampler": sampler_config['name']})
    
    return model_and_diffusion_config, task_config, sampler_config  


def main(model_and_diffusion_config, task_config, sampler_config, output_dir='./outputs', 
         sampler='tds', P=None, T=None, var_type=6, tausq=0.12, ess_threshold=None, t_truncate_percent=0, 
         classifier_guidance_scale=1.0, 
         class_batch_size=10, class_label=None, 
         mask=None,  
         dataset_label=None, dataset_offset=0, dataset_size=None, seed=42, 
         classifier_eval=False, 
         debug_plot=False, debug_statistics=False, save_input=False, save_output_i=10, 
         debug_first_i=0, 
         debug_plot_interval=10, debug_plot_start=-1, debug_plot_end=-1,  
         ):
    
    if sampler == 'replacement':
        ess_threshold = 0 
        sampler_is_smc_method = False 
    else:
        sampler_is_smc_method = True 

        
    model_and_diffusion_config, task_config, sampler_config = \
        process_configs(model_and_diffusion_config, task_config, sampler_config,  
                        sampler, P, T, var_type, tausq, ess_threshold, t_truncate_percent, \
                            mask, classifier_guidance_scale)
        
    sampler_name = sampler_config['name']
        
    T = model_and_diffusion_config["timestep_respacing"]
    P = sampler_config['num_particles']
    sampler_full_name = f"{sampler_name}_T{T}_P{P}_essthreshold{sampler_config['ess_threshold']}_truncate{sampler_config['t_truncate_percent']}"
    
    if task_config['name'] == 'class_cond_gen':
        classifier_guidance_scale = task_config['classifier_guidance_scale'] # TO add this 
        sampler_full_name = f"{sampler_full_name}/guidancescale{classifier_guidance_scale}"
    else:
        sampler_full_name = f"{sampler_full_name}/vartype{task_config.get('pred_xstart_var_type')}_tausq{task_config.get('tausq')}"
    

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if task_config['name'] == 'inpainting':
        output_path = os.path.join(output_dir, f"{task_config['name']}_{task_config['mask']}", f"{sampler_full_name}_data{dataset_label}-{dataset_offset}-{dataset_size}_{timestr}")    
    elif task_config['name'] == 'class_cond_gen':
        output_path = os.path.join(output_dir, f"{task_config['name']}", f"{sampler_full_name}_data{dataset_label}-{dataset_offset}-{dataset_size}_{timestr}")    
    else:
        raise NotImplementedError("Task undefined.") 
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"Saving to {output_path}", flush=True) 

    output_config_file = os.path.join(output_path, 'config.yml') 
    config_dict = {"seed": seed, 
                   "model_and_diffusion_config": model_and_diffusion_config,  
                   "task_config": task_config, 
                   "sampler_config": sampler_config, 
                   "dataset_label": dataset_label, 
                   "dataset_offset": dataset_offset, 
                   "dataset_size": dataset_size,  
                   "output_path": output_path}
    with open(output_config_file, 'w') as out:
        yaml.dump(config_dict, out, default_flow_style=False)
    
    model_and_diffusion_config = NoneDict(**model_and_diffusion_config)
    task_config = NoneDict(**task_config)
    sampler_config = NoneDict(**sampler_config)  

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    
    os.makedirs(os.path.join(output_path, "samples"))
    
    device = dist_util.dev(None)

    model, diffusion = create_model_and_diffusion(
        **select_args(model_and_diffusion_config, model_and_diffusion_defaults().keys())
        )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            model_and_diffusion_config.model_path), map_location="cpu")
    )
    model.to(device)

    if model_and_diffusion_config.use_fp16:
        model.convert_to_fp16()
    model.eval()
    # freeze model parameters and no grad 
    for param in model.parameters():
        param.requires_grad = False

    def model_fn(x, t, y=None, **kwargs):
        return model(x, t, y if model_and_diffusion_config.class_cond else None, **kwargs)


    if task_config.name == 'class_cond_gen':
        classifier = load_classifier(task_config['classifier_path'], device=device)

        classifier_prob_fn =  ConditioningMethod(operator=None).classifier_prob
        classifier_prob_fn = partial(classifier_prob_fn, classifier=classifier, guidance_scale=classifier_guidance_scale) 

    else:
        operator = get_operator(device=device, name=task_config.operator)   
        recon_prob_fn = ConditioningMethod(operator=operator).recon_prob
        diffusion.tausq_ = task_config.tausq 
        if classifier_eval:
            classifier = load_classifier(task_config['classifier_path'], device=device)


    diffusion.task = task_config.name 
    diffusion.use_mean_pred = True  
    diffusion.t_truncate = int(diffusion.T * t_truncate_percent)

    if task_config.name == 'class_cond_gen':
        if class_label is None:
            num_classes = 10 
            dl = np.tile(np.arange(num_classes), class_batch_size) # class labels
        else:
            dl = np.ones(class_batch_size, dtype=np.int64) * class_label 

    else:
        batch_size = 1
        dl = get_dataloader(task_config.dataset_path, batch_size=batch_size, 
                            image_size=task_config.image_size, 
                            class_cond=task_config.class_cond, 
                            rgb=task_config.rgb, 
                            return_dataloader=True, 
                            return_dict=True,
                            deterministic=True, 
                            dataset_label=dataset_label, 
                            offset=dataset_offset,
                            dataset_size=dataset_size) # no shuffling  

    if debug_plot or debug_statistics or save_input:
        debug_path = os.path.join(output_path, "debug")
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        assert debug_plot_end <= debug_plot_start
        debug_info = {"debug_path": debug_path, 
                      "plot_interval": debug_plot_interval, 
                      "plot_start": debug_plot_start,
                      "plot_end": debug_plot_end} 
    else:
        debug_info = {} 

    sample_path = os.path.join(output_path, "samples")
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    
    for i, data_i in enumerate(iter(dl)):
        if i > debug_first_i:
            debug_plot = debug_statistics = False 

        if task_config.name == 'class_cond_gen':
            class_label = data_i 
            ref_img_name = f"label{class_label}_iter{i}"
        else:
            ref_img_dict = data_i 
            ref_img = ref_img_dict['GT'].squeeze(0).to(device) 
            ref_img_name = ref_img_dict['GT_name'][0].split(".")[0] 
                
        debug_info.update({"img_name": ref_img_name}) 

        # just a placeholder since we use unconditional diffusion models 
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(), device=device
        )
        model_kwargs["y"] = classes
        

        # re-setting twisting functions 
        if task_config.name == 'inpainting':
            mask = gen_mask(task_config.mask, ref_img, ref_img_name)
            if len(mask.shape) == 4: measurement_mask = mask[0] # first dimension is extra degree-of-freedom 
            else: measurement_mask = mask
            measurement = operator(ref_img, mask=measurement_mask) # returns a one-dimensional tensor 
            assert measurement_mask.shape == ref_img.shape 
            
            recon_prob_fn = partial(recon_prob_fn, measurement=measurement, mask=mask)
            
            diffusion.mask = mask 
            diffusion.set_measurement(ref_img*measurement_mask) 
            # resetting 
            diffusion.recon_prob_fn = recon_prob_fn 
        
        elif task_config.name == 'class_cond_gen':
            classifier_prob_fn = partial(classifier_prob_fn, measurement=class_label)
            # resetting 
            diffusion.classifier_prob_fn = classifier_prob_fn 

        else:
            raise NotImplementedError 
        
        if task_config.name != 'class_cond_gen' and i < 10 and save_input:
            imwrite(os.path.join(debug_path, f'{ref_img_name}_measurement.png'), toU8(ref_img*measurement_mask))
            imwrite(os.path.join(debug_path, f"{ref_img_name}_gt.png"), toU8(ref_img))
    
        diffusion.clear_cache() 
        
        M = partial(diffusion.M, model=model_fn, device=device, 
                    pred_xstart_var_type=task_config.pred_xstart_var_type) 
        G = partial(diffusion.G, model=model_fn, 
                    debug_plot=debug_plot, debug_statistics=debug_statistics, debug_info=debug_info, 
                    pred_xstart_var_type=task_config.pred_xstart_var_type)
        
        # Sampling 
        final_sample, log_w, normalized_w, resample_indices_trace, ess_trace, log_w_trace, xt_trace  = \
            smc_FK(M=M, G=G, 
                   resample_strategy=sampler_config.resample_strategy, 
                   ess_threshold=sampler_config.ess_threshold, 
                   T=diffusion.T, 
                   P=sampler_config.num_particles, 
                   verbose=True, 
                   log_xt_trace=False, 
                   extra_vals={"model_kwargs": model_kwargs, 
                               "batch_p": sampler_config.batch_p})


        # Gathering results, plotting and saving 
        if diffusion.t_truncate >1:
            truncate_sample = diffusion.xpred_at_t_truncate.cpu()
            truncate_log_w = diffusion.log_w_x0_truncate.cpu()

            if sampler_config.ess_threshold == 0:
                log_w_Ttotp1 = log_w_trace[-(diffusion.t_truncate+2)]
                assert log_w_Ttotp1.shape == truncate_log_w.shape, f"{log_w_Ttotp1.shape}, {truncate_log_w.shape}"
                truncate_log_w += log_w_Ttotp1 
                
        if i < save_output_i:
            print("\nSaving!", flush=True)
            plt.plot(ess_trace.numpy()[::-1]) #(T+1, )
            plt.xlabel("t")
            plt.ylabel("ess")
            plt.ylim([0-1, sampler_config.num_particles+1])
            plt.title("Effective Sample Size")
            plt.savefig(os.path.join(sample_path, f"{ref_img_name}_ess.png"))
            plt.close()

            nrow = int(np.ceil(np.sqrt(sampler_config.num_particles)))
            gridded_image = make_grid(final_sample, nrow) # (C, H_new, W_hew)
            imwrite(path=os.path.join(sample_path, f'./sample_{ref_img_name}.png'), img=toU8(gridded_image)) # np arr (H, W, C)

            if sampler_is_smc_method:
                most_likely_idx = th.argmax(normalized_w)
                most_likely_sample = final_sample[most_likely_idx]
                imwrite(path=os.path.join(sample_path, f'./sample_{ref_img_name}_most_likely.png'), img=toU8(most_likely_sample)) 

            if diffusion.t_truncate > 1:
            
                gridded_image = make_grid(truncate_sample, nrow) # (C, H_new, W_hew)
                imwrite(path=os.path.join(sample_path, f'./sample_truncate_{ref_img_name}.png'), img=toU8(gridded_image)) # np arr (H, W, C)
                
                if sampler_is_smc_method:
                    most_likely_idx = th.argmax(truncate_log_w)
                    most_likely_sample = truncate_sample[most_likely_idx]
                    imwrite(path=os.path.join(sample_path, f'./sample_truncate_{ref_img_name}_most_likely.png'), img=toU8(most_likely_sample)) 


        save_dict = dict(sample=final_sample.cpu().numpy(), 
                         ess_trace=ess_trace.numpy()[::-1], 
                         log_w=log_w.cpu().numpy())

        if task_config.name == "class_cond_gen":
            save_dict.update(classifier_logprob_x0=diffusion.classifier_logprob_x0.cpu().numpy())

        else:
            if classifier_eval:
                # evaluate the prob given classifier 
                x0_logits = pred(classifier, final_sample, return_probs=False) # (P, num_classes)
                save_dict.update(classifier_x0_logits=x0_logits)
                

        if diffusion.t_truncate > 1:
            save_dict.update(
                log_w_x0_truncate=truncate_log_w.numpy(), 
                xpred_at_t_truncate=diffusion.xpred_at_t_truncate.cpu().numpy(), 
                ess_truncate = compute_ess_from_log_w(truncate_log_w).numpy()
                )
            
            if task_config.name == "class_cond_gen":
                save_dict.update(classifier_logprob_x0truncate=diffusion.classifier_logprob_x0truncate.cpu().numpy())

            else:
                if classifier_eval:
                    # evaluate the prob given classifier 
                    x0_truncate_logits = pred(classifier, diffusion.xpred_at_t_truncate, return_probs=False)
                    save_dict.update(classifier_x0_truncate_logits=x0_truncate_logits)

        np.savez_compressed(os.path.join(sample_path, f"{ref_img_name}_sample.npz"),
                            **save_dict)



if __name__ == "__main__":

    fire.Fire(main)
