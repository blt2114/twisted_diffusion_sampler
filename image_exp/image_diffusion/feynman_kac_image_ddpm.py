# Adapted from https://github.com/openai/guided-diffusion, which is under the MIT license


import torch  as th 
from torch.distributions import Normal 
from .smc_diffusion import SMCDiffusion 
from .image_util import multi_imgwrite, toU8, imwrite, visualize_weights  
import os 
from torchvision.utils import make_grid 
import numpy as np 


def log_normal_density(sample, mean, var):
    return Normal(loc=mean, scale=th.sqrt(var)).log_prob(sample)
 


__SMC_DDPM__ = {} 

def register_smc_ddpm(name: str):
    def wrapper(cls):
        if __SMC_DDPM__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SMC_DDPM__[name] = cls
        return cls
    return wrapper

def get_smc_ddpm(name: str, **kwargs):
    if __SMC_DDPM__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SMC_DDPM__[name](**kwargs)


@register_smc_ddpm(name='unconditional')
class UnconditionalDDPM(SMCDiffusion):
    # for unconditional sampling 
    def M(self, t, xtp1, extra_vals, P, model, device, collapse_proposal_shape=True, **kwargs):

        if t == self.T: 
            xt = self.ref_sample(P, device=device)
        else:
            out = self.p_trans_model(xtp1=xtp1, t=t, model=model, clip_denoised=True, 
                                     model_kwargs=extra_vals.get('model_kwargs', None))
            xt_mean = out["mean_untwisted"]
            xt_var = out["var_untwisted"]
            xt = xt_mean + (t!=0)*th.sqrt(xt_var) * th.randn_like(xt_mean)

            if self.t_truncate > 1 and t == self.t_truncate:
                self.xpred_at_t_truncate = out["pred_xstart"]
                extra_vals['xt_mean'] = xt_mean
                extra_vals['xt_var'] = xt_var 

            if t == 0 and self.task == 'inpainting':
                extra_vals['xt_mean'] = xt_mean
                extra_vals['xt_var'] = xt_var 

        return xt, extra_vals 

    def G(self, t, xtp1, xt, extra_vals, model, debug_plot=False, debug_info=None, **kwargs):
        P = xt.shape[0]
        return th.zeros(P)


@register_smc_ddpm(name='is')
class NaiveDDPM(UnconditionalDDPM):
    # Importance sampling, i.e. using unconditional model as proposal     
    def M(self, t, xtp1, extra_vals, P, model, device, collapse_proposal_shape=True, **kwargs):
        
        xt, extra_vals = super().M(t, xtp1, extra_vals, P, model, device, collapse_proposal_shape) 

        if self.t_truncate > 1 and t == self.t_truncate and self.task == 'inpainting':
            mask = self.mask.float()  
            self.xpred_at_t_truncate = self.measurement * mask + self.xpred_at_t_truncate * (1-mask) 

        if t == 0 and self.task == 'inpainting':
            mask = self.mask.float()  
            xt_M = self.measurement  
            xt = xt_M * mask + xt * (1-mask) 
        
        return xt, extra_vals 
        
    
    def G(self, t, xtp1, xt, extra_vals, model, debug_plot=False, debug_info=None, **kwargs):
        P = xt.shape[0] 

        if self.t_truncate > 1 and t == self.t_truncate:
            if self.task == 'inpainting':
                xt_mean = extra_vals.pop("xt_mean")
                xt_var = extra_vals.pop("xt_var")
                log_w_x0_truncate =  log_normal_density(xt, xt_mean, xt_var) 
                self.log_w_x0_truncate = (self.mask.float() * log_w_x0_truncate).sum(dim=self.particle_base_dims) # (P, ) 
            
            elif self.task == 'class_cond_gen':
                logp_y_given_x0, classifier_logprob_x0truncate \
                    = self.classifier_prob_fn(\
                    xt=None, x0_hat=self.xpred_at_t_truncate, \
                        return_grad=False, return_full_prob_original_scale=True) 
                self.log_w_x0_truncate = logp_y_given_x0 
                
                self.classifier_logprob_x0truncate = classifier_logprob_x0truncate  
            else:
                raise ValueError("Unsupported task: ", self.task)

        if t == 0:
            if self.task == 'inpainting':
                xt_mean = extra_vals.pop("xt_mean")
                xt_var = extra_vals.pop("xt_var")
                log_w =  log_normal_density(xt, xt_mean, xt_var) 
                log_w = (self.mask.float() * log_w).sum(dim=self.particle_base_dims) # (P, ) 

            elif self.task == 'class_cond_gen':
                classifier_logprob_y_given_x0, classifier_logprob_x0 \
                    = self.classifier_prob_fn(xt=None, x0_hat=xt, \
                                              return_grad=False, return_full_prob_original_scale=True)
                log_py_given_x0 = classifier_logprob_y_given_x0 # (P,)
                assert log_py_given_x0.shape == (P,), log_py_given_x0.shape 
                
                #log_target =  log_py_given_x0 
                log_w = log_py_given_x0

                self.classifier_logprob_x0 = classifier_logprob_x0  
            else:
                raise ValueError("Unsupported task: ", self.task)
        
        else:
            log_w = th.zeros(P, device=xt.device, dtype=xt.dtype)

        return log_w 


@register_smc_ddpm(name="tds")
class TwistedDDPM(SMCDiffusion):

    def M(self, t, xtp1, extra_vals, P, model, device, collapse_proposal_shape=True, pred_xstart_var_type=1):
        
        if t == self.T:
            self.clear_cache() 

            xt = self.ref_sample(P, device=device)
            log_proposal = self.ref_log_density(xt)
            if collapse_proposal_shape:
                log_proposal = log_proposal.sum(self.particle_base_dims)

        else:
            # For t >= 1, sample from proposal p(x_t | x_{t+1}, y) 
            # For t = 0, sample from model p(x_0 | x_1)
            # the proposal distribution is precomputed in psi(x_{t+1}) from previous iteration

            resample_idcs = extra_vals[("resample_idcs", t+1)]
            xt_var = self.cache.pop(("xt_var", t))[resample_idcs] #  std at t =0 is around 0.008
            xt_mean = self.cache.pop(("xt_mean", t))[resample_idcs]

            if self.use_mean_pred:
                xt = xt_mean + (t!=0) * th.randn_like(xt_mean) * th.sqrt(xt_var)
            else:
                xt = xt_mean + th.randn_like(xt_mean) * th.sqrt(xt_var)
            assert not xt.requires_grad 
            
            log_proposal = log_normal_density(xt, xt_mean, xt_var)
            if collapse_proposal_shape:
                log_proposal = log_normal_density(xt, xt_mean, xt_var).sum(self.particle_base_dims)
            if self.use_mean_pred:
                log_proposal = (t!=0) * log_proposal 

        if t == 0 and self.task == 'inpainting':
                mask = self.mask.float() 
                assert mask.shape == xt.shape[1:], mask.shape # (C, H, W)
                y = self.measurement 
                assert y.shape == xt.shape[1:], y.shape 
                xt = y * mask + xt * (1-mask) 
           
        self.cache[('log_proposal', t)] = log_proposal
        return xt, extra_vals

    def G(self, t, xtp1, xt, extra_vals, model, debug_plot=False, debug_statistics=False, debug_info=None, pred_xstart_var_type=1):
        P = xt.shape[0]
        assert xt.shape[1:] == self.particle_base_shape, xt.shape 

        log_proposal = self.cache.pop(('log_proposal', t))

        #######################################################
        # gathering previous and current log_potential values #
        #######################################################
        if t == self.T:
            log_potential_xtp1 = th.zeros_like(log_proposal)     
            log_p_trans_untwisted = log_proposal  

            if debug_statistics:
                self.debug_statistics = {'resample_idcs_tp1': [], 
                                        't': [],
                                        'log_target': [], 
                                        'log_p_trans_untwisted':[], 
                                        'log_potential_xt': [], 
                                        'log_potential_xtp1': [], 
                                        'log_proposal': [], 
                                        'log_w': []}

        else: 
            # resample the cached values at previous iteration 
            resample_idcs = extra_vals.pop(("resample_idcs", t+1))
            
            if debug_statistics:
                self.debug_statistics['resample_idcs_tp1'].append(resample_idcs.detach().cpu())
            
            log_potential_xtp1 = self.cache.pop(('log_potential', t+1)) # (P, C, H, W)
            log_potential_xtp1 = log_potential_xtp1[resample_idcs] 
            
            xt_mean_untwisted = self.cache.pop(('xt_mean_untwisted', t))[resample_idcs]
            xt_var = self.cache.pop(('xt_var_untwisted', t))[resample_idcs]
            log_p_trans_untwisted = log_normal_density(xt, xt_mean_untwisted, xt_var)

            if t == 0:
                # log p(y|x1)
                if self.task == 'inpainting':
                    # note that x0_M is set to y in the proposal 
                    log_p_trans_untwisted_y = (log_p_trans_untwisted * self.mask).sum(dim=self.particle_base_dims)

            log_p_trans_untwisted = log_p_trans_untwisted.sum(dim=self.particle_base_dims)


        #######################################################
        # calculating log_potential_t and the proposal at t-1 #
        #######################################################
        
        if t > 0: 
            batch_p = extra_vals.get("batch_p", P) 
            xtm1_mean, log_potential_xt, mean_untwisted, var_untwisted, \
                pred_xstart, twisted_pred_xstart, grad_log_potential_xt = \
                self._compute_twisted_step(batch_p, xt, t, 
                                            model=model, 
                                            model_kwargs=extra_vals.get('model_kwargs', None), 
                                            debug_plot=debug_plot, 
                                            pred_xstart_var_type=pred_xstart_var_type)
            
            self.cache[('xt_mean_untwisted', t-1)] = mean_untwisted
            self.cache[('xt_var_untwisted', t-1)] = var_untwisted
            if t == 1:
                self.cache[("xt_mean", t-1)] = mean_untwisted 
                self.cache[("xt_var", t-1)] = var_untwisted
            else:
                self.cache[("xt_mean", t-1)] = xtm1_mean 
                self.cache[("xt_var", t-1)] = var_untwisted
            self.cache[('log_potential', t)] = log_potential_xt

            log_target = log_p_trans_untwisted + log_potential_xt - log_potential_xtp1  
            log_w = log_target - log_proposal

            if self.t_truncate > 1 and t == self.t_truncate:
                pred_xstart = pred_xstart.to(xt.device)
                    
                if self.task == 'inpainting':
                    # make one-step prediction of x0 NOTE: twisted or not twisted? 
                    mask = self.mask.float()
                    self.xpred_at_t_truncate = self.measurement * mask + pred_xstart * (1-mask) 

                    # calculate weight for (xt, x0^M | xtp1)
                    #pred_var = self._model_variance_at_t_truncate.detach()
                    pred_var = th.ones_like(pred_xstart)*(1 - self.alphas_cumprod[t])
                    logp_y_given_xt = log_normal_density(self.measurement, pred_xstart, pred_var)
                    assert logp_y_given_xt.shape == xt.shape, logp_y_given_xt.shape 
                    logp_y_given_xt = (logp_y_given_xt * mask).sum(self.particle_base_dims)
                    # logp(xt|xtp1) + logp(x0^M|xt) - log_potential_xtp1 - log_proposal(xt|xtp1)
                    self.log_w_x0_truncate = log_p_trans_untwisted + logp_y_given_xt - log_potential_xtp1 - log_proposal

                elif self.task == 'class_cond_gen':
                    self.xpred_at_t_truncate = pred_xstart

                    # logtarget(x0,xt|xtp1) = delta(x0|\hat(xt) + logp(y|x0) + logp(xt|xtp1) - log_potential_xtp1
                    # logproposal(x0,xt|xtp1) = delta(x0|\hat(xt)) + log_proposal(xt|xtp1,y)
                     
                    logp_y_given_x0, classifier_logprob_x0truncate \
                        = self.classifier_prob_fn(xt=None, x0_hat=pred_xstart, 
                                                  return_grad=False, return_full_prob_original_scale=True) 
                    self.log_w_x0_truncate = logp_y_given_x0 + log_p_trans_untwisted - log_potential_xtp1 - log_proposal 
                    
                    self.classifier_logprob_x0truncate = classifier_logprob_x0truncate # a hack for now 
    
                pred_xstart = pred_xstart.cpu() 
                
        else:
            # t = 0 
            if self.task == 'inpainting':
                # weight_0 = p_trans_untwisted(y|x_1) / p_trans_twisted(y|x_1)
                log_p_trans_twisted_y = log_potential_xtp1 
                log_w = log_p_trans_untwisted_y - log_p_trans_twisted_y 
                log_target = th.zeros_like(log_w) # placeholder 
            elif self.task == 'class_cond_gen':
                classifier_logprob_y_given_x0, classifier_logprob_x0  \
                    = self.classifier_prob_fn(xt=None, x0_hat=xt, \
                                              return_grad=False, return_full_prob_original_scale=True)
                log_py_given_x0 = classifier_logprob_y_given_x0 # (P,)
                assert log_py_given_x0.shape == (P,), log_py_given_x0.shape 
                
                # suppose using mean prediction in the final step so proposal and diffusion target canceld out
                log_target =  log_py_given_x0 - log_potential_xtp1  
                log_w = log_target

                self.classifier_logprob_x0 = classifier_logprob_x0  #hack for now

        if debug_statistics:
            self.debug_statistics['t'].append(t)
            self.debug_statistics['log_target'].append(log_target.detach().cpu())
            self.debug_statistics['log_p_trans_untwisted'].append(log_p_trans_untwisted.detach().cpu())
            self.debug_statistics['log_potential_xt'].append(log_potential_xt.detach().cpu())
            self.debug_statistics['log_potential_xtp1'].append(log_potential_xtp1.detach().cpu())
            self.debug_statistics['log_proposal'].append(log_proposal.detach().cpu())
            self.debug_statistics['log_w'].append(log_w.detach().cpu())
        
        if debug_plot and t > 0:  
            debug_path = debug_info.get("debug_path")
            img_name = debug_info.get("img_name") 
            plot_interval = debug_info.get("plot_interval")
            plot_start = debug_info.get("plot_start")
            plot_end = max(debug_info.get("plot_end", 0), 1)

            if (t % plot_interval == 0 and t < plot_start and t >= plot_end):
                # pred_xstart: (P, C, H, W) 
                # --make_grid -> (C, H_new, W_new)
                nrow = int(np.ceil(np.sqrt(P)))  
                save_type = 'png'
                imwrite(os.path.join(debug_path, f'{img_name}_t{t}_predxstart.{save_type}'), 
                            toU8(make_grid(pred_xstart, nrow)))
                imwrite(os.path.join(debug_path, f"{img_name}_t{t}_predxstart_twisted.{save_type}"), 
                            toU8(make_grid(twisted_pred_xstart, nrow)))
                
                visualize_weights(os.path.join(debug_path, f'{img_name}_t{t}_weight.{save_type}'), log_w.detach().cpu())
                
        
        if debug_statistics and t == 0:
            debug_path = debug_info.get("debug_path")
            img_name = debug_info.get("img_name")
            for key, val in self.debug_statistics.items():
                if key != 't':
                    self.debug_statistics[key] = th.stack(val,dim=0)
            th.save(self.debug_statistics, os.path.join(debug_path, f'{img_name}_t{t}.pkl'))
            self.debug_statistics = {} 
        
        assert log_w.shape == (P, ), f"t={t}, {log_w.shape}" 

        return log_w 
    
    def _compute_twisted_step(self, batch_p, xt, t, model, model_kwargs, debug_plot=False, 
            pred_xstart_var_type=1, target_pred_xstart_var_type=None):
        """compute xtm1_mean and xtm1_var given xt after applying the twisted operation""" 
        P = xt.shape[0]
        beta_t = self.betas[t-1]

        # do a loop here in case of memory overflow 
        
        alphas_cumprod_t = self.alphas_cumprod[t-1] # python indexing

        def get_xstart_var(var_type):
            
            sigmasq_ = (1-alphas_cumprod_t) / alphas_cumprod_t
            if var_type == 1:
                return sigmasq_ 
            elif var_type == 2: # pseudoinverse-guided paper https://openreview.net/forum?id=9_gsMA8MRKQ 
                tausq_ = 1.0 
                return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
                #return (1 - alphas_cumprod_t) 
            elif var_type == 5: 
                tausq_ = 0.30 
                return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
            elif var_type == 3: # DPS paper https://arxiv.org/abs/2209.14687 
                return None  
            elif var_type == 4: # pseudoinverse-guided paper -- the actual implementation, see their Alg.1 
                return beta_t  / np.sqrt(alphas_cumprod_t) 
            elif var_type == 6: # freely specify tausq_
                tausq_ = self.tausq_ 
                return (sigmasq_ * tausq_) / (sigmasq_ + tausq_)
        
        if self.task == 'inpainting':
            pred_xstart_var = get_xstart_var(pred_xstart_var_type)
            if target_pred_xstart_var_type is None:
                target_pred_xstart_var = pred_xstart_var
            else:
                target_pred_xstart_var = get_xstart_var(target_pred_xstart_var_type)

        xt_batches = th.split(xt.cpu(), batch_p)
        
        mean_untwisted = []
        var_untwisted = [] 
        pred_xstart = []  
        grad_log_potential_xt = [] 
        twisted_pred_xstart = [] 
        log_potential_xt = [] 
        xtm1_mean = [] 
        for xt_batch in xt_batches:
            xt_batch = xt_batch.to(xt.device).requires_grad_() 
            out = self.p_trans_model(xtp1=xt_batch, t=t-1, model=model, clip_denoised=False, 
                                    model_kwargs=model_kwargs)
            
            if self.task == 'class_cond_gen':
                log_potential_xt_batch, grad_log_potential_xt_batch = self.classifier_prob_fn(xt=xt_batch, x0_hat=out['pred_xstart'])
                assert grad_log_potential_xt_batch.shape == xt_batch.shape 
                assert log_potential_xt_batch.shape == (P,)

            else:
                # reconstruction guidance                 
                # compute -0.5 * ||f(x0) - f(x0_hat)||^2
                recon_prob, recon_prob_grad = self.recon_prob_fn(xt=xt_batch, x0_hat=out['pred_xstart'])
                assert recon_prob_grad.shape == xt_batch.shape  # (P_batch, C, H, W)

                # compute log_potential and log_potential_grad 
                if pred_xstart_var_type == 3:
                    norm = th.sqrt(-2*recon_prob.detach())
                    guidance_scale = 0.5 
                    pred_xstart_var = norm * beta_t / guidance_scale
                    assert pred_xstart_var.shape == (xt_batch.shape[0],), pred_xstart_var.shape
                    if target_pred_xstart_var_type is None:
                        target_pred_xstart_var = pred_xstart_var 

                    grad_log_potential_xt_batch = 1./pred_xstart_var[:, None, None, None] * recon_prob_grad
                else:
                    grad_log_potential_xt_batch = 1./pred_xstart_var * recon_prob_grad 
                log_potential_xt_batch = 1./target_pred_xstart_var * recon_prob  

            mean_untwisted.append(out['mean_untwisted'].detach())
            var_untwisted.append(out['var_untwisted'].detach())
            pred_xstart.append(out['pred_xstart'].detach().cpu()) 
            if debug_plot:
                grad_log_potential_xt.append(grad_log_potential_xt_batch.cpu())
            log_potential_xt.append(log_potential_xt_batch)

            xt_batch.requires_grad_(False)
            assert not log_potential_xt_batch.requires_grad 
            assert not grad_log_potential_xt_batch.requires_grad  

            #xtm1_mean_batch = out['mean_untwisted'].detach() + beta_t * grad_log_potential_xt_batch 
            xtm1_mean_batch, twisted_pred_xstart_batch = self._compute_twisted_step_helper(
                xt=xt_batch, 
                t=t, 
                untwisted_pred_xstart=out['pred_xstart'].detach(), 
                grad_log_potential=grad_log_potential_xt_batch, 
                return_twisted_pred_xstart=debug_plot
            )
            xtm1_mean.append(xtm1_mean_batch) 
            
            if debug_plot:
                twisted_pred_xstart.append(twisted_pred_xstart_batch.cpu())

        return [safe_cat(lt) for lt in [xtm1_mean, log_potential_xt, mean_untwisted, var_untwisted, \
                                          pred_xstart, twisted_pred_xstart, grad_log_potential_xt]]
    
    def _compute_twisted_step_helper(self, xt, t, untwisted_pred_xstart, grad_log_potential, 
                                     return_twisted_pred_xstart=False, 
                                     denoised_fn=None, clip_twisted=True):
        """compute twisted mean for xtm1 given xt"""
        alphas_cumprod_t = self.alphas_cumprod[t-1]
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t-1]
        twisted = (1-alphas_cumprod_t) / (sqrt_alphas_cumprod_t) * grad_log_potential 
        twisted_pred_xstart = untwisted_pred_xstart + twisted
        
        def process_xstart(x): 
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_twisted:
                return x.clamp(-1, 1)
            return x

        twisted_pred_xstart = process_xstart(twisted_pred_xstart)

        xtm1_mean, _, _ = self.q_posterior_mean_variance(
                x_start=twisted_pred_xstart, 
                x_t=xt, 
                t=t-1 
        )
        if return_twisted_pred_xstart:
            return xtm1_mean, twisted_pred_xstart.cpu()  
        return xtm1_mean, None 
    

"""
Impelmentation of SMCDiff from the paper 
Diffusion probabilistic modeling of protein backbones in 3D for the motif-scaffolding problem
https://arxiv.org/abs/2206.04119 

Right now only consider inpainting problem
"""
@register_smc_ddpm(name='smcdiff')
class SMCDiffDDPM(SMCDiffusion):

    def M(self, t, xtp1, extra_vals, P, model, device, **kwargs) :
        # apply replacement proposal 
        mask = self.mask.float() 

        if t == self.T:
            self.clear_cache() 

            xts_forward_given_y = self._sample_forward(x0=self.measurement) # shape (T+1, C, H, W)
            self.cache["xts_forward_given_y"] = xts_forward_given_y 
            xt_S = self.ref_sample(P, device=device)

            xt_mean = th.zeros_like(xt_S)
            xt_var = th.ones_like(xt_S)
        else:
            xts_forward_given_y = self.cache['xts_forward_given_y']

            out = self.p_trans_model(xtp1=xtp1, t=t, model=model, clip_denoised=True, 
                                     model_kwargs=extra_vals.get('model_kwargs', None))
            xt_mean = out["mean_untwisted"]
            xt_var = out["var_untwisted"]
            xt_S = xt_mean + (t!=0)*th.sqrt(xt_var) * th.randn_like(xt_mean)

            self.cache[('pred_xstart', t)] = out["pred_xstart"] # for debugging purpose 

            if self.t_truncate > 1 and t == self.t_truncate: 
                # make one-step prediction of x0 
                pred_xstart = out['pred_xstart']
                self._pred_xstart_at_t_truncate = pred_xstart
                self.xpred_at_t_truncate = self.measurement * mask + pred_xstart * (1-mask) 

        
        xt_M = xts_forward_given_y[t].to(device) 

        assert mask.shape == xt_S.shape[1:] # (C, H, W)
        xt = xt_M * mask + xt_S * (1-mask) 

        self.cache[('xt_mean', t)] = xt_mean 
        self.cache[('xt_var', t)] = xt_var 

        return xt, extra_vals

    def G(self, t, xtp1, xt, extra_vals, model, debug_plot=False, debug_info=None, **kwargs):
        # log_w = logp(x_t^S, x_t^M | x_tp1) - logp(x_t^S|x_tp1) = logp(x_t^M |x_tp1)
        mask = self.mask.float()

        xt_mean = self.cache.pop(('xt_mean', t))
        xt_var = self.cache.pop(('xt_var', t))
        log_w = log_normal_density(xt, xt_mean, xt_var) 
        log_w = (mask * log_w).sum(dim=self.particle_base_dims) # (P, )
        self.cache[('log_w', t)] = log_w 

        if self.t_truncate > 1 and t == self.t_truncate: 
            # calculate weight for (xt, x0^M | xtp1)
            #pred_var = self._model_variance_at_t_truncate.detach()
            pred_xstart = self._pred_xstart_at_t_truncate
            pred_var = th.ones_like(pred_xstart)*(1 - self.alphas_cumprod[t])
            logp_y_given_xt = log_normal_density(self.measurement, pred_xstart, pred_var)
            assert logp_y_given_xt.shape == xt.shape, logp_y_given_xt.shape 
            logp_y_given_xt = (logp_y_given_xt * mask).sum(self.particle_base_dims)
            # logp(xt|xtp1) + logp(x0^M|xt) - log_potential_xtp1 - log_proposal(xt|xtp1)
            self.log_w_x0_truncate = logp_y_given_xt 
        
        if t <self.T:
            pred_xstart = self.cache.pop(('pred_xstart', t))
        if debug_plot: 
            debug_path = debug_info.get("debug_path")
            img_name = debug_info.get("img_name") 
            plot_interval = debug_info.get("plot_interval")
            plot_start = debug_info.get("plot_start")

            if (t % plot_interval == 0 and t < plot_start and t < self.T) or  t == 1:
                P = xt.shape[0] 
                nrow = int(np.sqrt(P))   
                multi_imgwrite(
                    save_path=os.path.join(debug_path, f'{img_name}_t{t}.png'), 
                    img_list=[make_grid(xt.cpu(), nrow)[0].numpy(), 
                            make_grid(pred_xstart.cpu(), nrow)[0].numpy()], 
                    img_names=['xt', 'pred_xstart'], 
                    title=f'{img_name}_t{t}', 
                    vmin=-1, vmax=1
                )    

        assert log_w.shape == (xt.shape[0], ), log_w.shape
        
        return log_w 


# Not really a SMC method but will keep it under the name 
@register_smc_ddpm(name='replacement')
class ReplacementDDPM(SMCDiffusion):

    def M(self, t, xtp1, extra_vals, P, model, device, **kwargs) :
        # apply replacement proposal 
        mask = self.mask.float() 

        if t == self.T:

            self.clear_cache() 

            xts_forward_given_y = self._sample_forward(x0=self.measurement, batch_size=P) # shape (T+1, P, C, H, W)
            self.cache["xts_forward_given_y"] = xts_forward_given_y 
            xt_S = self.ref_sample(P, device=device)

            xt_mean = th.zeros_like(xt_S)
            xt_var = th.ones_like(xt_S)
        else:
            xts_forward_given_y = self.cache['xts_forward_given_y']

            out = self.p_trans_model(xtp1=xtp1, t=t, model=model, clip_denoised=True, 
                                     model_kwargs=extra_vals.get('model_kwargs', None))
            xt_mean = out["mean_untwisted"]
            xt_var = out["var_untwisted"]
            xt_S = xt_mean + (t!=0)*th.sqrt(xt_var) * th.randn_like(xt_mean)

            self.cache[('pred_xstart', t)] = out["pred_xstart"] # for debugging purpose 

            if self.t_truncate > 1 and t == self.t_truncate: 
                # make one-step prediction of x0 
                pred_xstart = out['pred_xstart']
                self._pred_xstart_at_t_truncate = pred_xstart
                self.xpred_at_t_truncate = self.measurement * mask + pred_xstart * (1-mask) 

                self.log_w_x0_truncate = th.zeros(P) # place holder 

        xt_M = xts_forward_given_y[t].to(device) 

        assert xt_M.shape == xt_S.shape, f"xt_M.shape={xt_M.shape}, xt_S.shape={xt_S.shape}"

        assert mask.shape == xt_S.shape[1:] # (C, H, W)
        xt = xt_M * mask + xt_S * (1-mask) 

        return xt, extra_vals

    def G(self, t, xtp1, xt, extra_vals, model, debug_plot=False, debug_info=None, **kwargs):
        P = xt.shape[0]
        return th.zeros(P)
    


def safe_cat(list_of_tensors, dim=0):
    if len(list_of_tensors) == 0:
        return None 
    return th.cat(list_of_tensors, dim=dim)
