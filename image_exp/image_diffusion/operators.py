# Adapted from https://github.com/DPS2022/diffusion-posterior-sampling


import torch as th 
from abc import ABC, abstractmethod 
from .feynman_kac_image_ddpm import log_normal_density 
import torch.nn.functional as F


#######################
# conditioning method #
#######################
# measurement_cond_fn is an instantiation of conditioning method with given measurement 

__CONDITIONING_METHOD__ = {} 

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, **kwargs)

# consider determinstic constraint for now  
class ConditioningMethod(ABC):
    def __init__(self, operator, **kwargs):
        self.operator = operator 

    def classifier_prob(self, xt, x0_hat, measurement, classifier, guidance_scale=1.0, return_grad=True, 
                        return_full_prob_original_scale=False):
        #using mean prediction, i.e. x0_hat for now 
        
        # guidance_scale: used to power the likelihood, i.e. pi = normalize(pi**2) 
        y = measurement # label 
        
        if return_grad:
            with th.enable_grad():
                logits = classifier(x0_hat)
                log_probs = F.log_softmax(guidance_scale*logits, dim=-1) # (P, num_classes)
                selected = log_probs[range(len(logits)), y] #(P, )
               
            return selected.detach(), th.autograd.grad(selected.sum(), xt)[0] 
        else:
            with th.no_grad():
                logits = classifier(x0_hat)
                log_probs = F.log_softmax(guidance_scale*logits, dim=-1)
                selected = log_probs[range(len(logits)), y]
            if return_full_prob_original_scale:
                with th.no_grad():
                    log_probs = F.log_softmax(logits, dim=-1)
                return selected, log_probs 
            return selected 


    
    def recon_prob(self, xt, x0_hat, measurement, **kwargs):
        assert len(x0_hat.shape) == 4
        P, C, H, W = x0_hat.shape
        base_shape = x0_hat.shape[1:]
        base_dims = [-(i+1) for i in range(len(base_shape))]

        mask = kwargs.get("mask", None)

        if mask is not None:
            assert len(mask.shape) == 3
            mask = mask.unsqueeze(0) # makes extra degree of freedom             

            y_mean_pred = self.operator(x0_hat, mask=mask)
            y_mean_pred = y_mean_pred.view(P, -1) # (P, measurement_dimensions) 
            assert y_mean_pred.shape[1:] == measurement.shape


            difference = measurement - y_mean_pred 
            recon_prob = -0.5 * th.sum(difference**2, dim=-1)
            assert recon_prob.shape == (P,), recon_prob.shape # (P, )
            recon_prob_grad = th.autograd.grad(recon_prob.mean(dim=0), xt)[0]
            recon_prob_grad *= P 
                
            return recon_prob.detach(), recon_prob_grad 

    def log_potential(self, xt, x0_hat, x0_var, measurement, return_grad=True, **kwargs):
        # y = measurement
        # h = operator 
        # p(y|x0_hat, x0_var) \approx N(y| h(x0_hat)), x0_var grad_h grad_h^T)
        # compute log p(y|x0_hat, x0_var) and grad_{xt} log p(y|x0_hat, x0_var) 

        assert len(xt.shape) == 4
        P, C, H, W = xt.shape
        base_shape = xt.shape[1:]
        base_dims = [-(i+1) for i in range(len(base_shape))]

        mask = kwargs.get("mask", None)
        if mask is not None:
            log_potential = [] 
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(0) # makes extra degree of freedom             

            for mask_i in mask:
                y_mean = self.operator(x0_hat, mask=mask_i)
                y_mean = y_mean.view(P, -1) # (P, measurement_dimensions) 
                assert y_mean.shape[1:] == measurement.shape

                if x0_var is None :
                    difference = measurement - y_mean.detach() 
                    norm = th.linalg.norm(difference,dim=-1) 
                    assert norm.shape == (P, ), norm.shape 
                    beta_t = kwargs.get("beta_t")
                    guidance_scale = kwargs.get("guidance_scale", 0.5)
                    x0_var = norm * beta_t / guidance_scale
                    assert x0_var.shape == (P,), x0_var.shape 
                    x0_var = x0_var[:, None]
                
                if isinstance(x0_var, th.Tensor):
                    assert not x0_var.requires_grad
                
                h_grad = self.operator.grad(x0_hat) 
                y_var = x0_var * h_grad**2
                    
                log_potential_i = log_normal_density(measurement, y_mean, y_var)
                log_potential_i = log_potential_i.sum(dim=-1)
                assert log_potential_i.shape == (P,), log_potential_i.shape                     

                log_potential.append(log_potential_i)
            log_potential = th.logsumexp(th.stack(log_potential, dim=0), dim=0) #(P, )
            assert log_potential.shape == (P,), log_potential.shape  
        else:
            raise NotImplementedError

        grad_log_potential = None 
        if return_grad:
            #grad_log_potential_ = th.autograd.grad(log_potential, xt, grad_outputs=th.ones_like(log_potential))[0] 
            grad_log_potential = th.autograd.grad(log_potential.mean(dim=0), xt)[0]
            grad_log_potential = grad_log_potential * P 
           
        return log_potential.detach(), grad_log_potential


############
# operator #
############
 
__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class Operator(ABC):

    def __call__(self, data, **kwargs):
        return self.forward(data, **kwargs)

    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate h(x)
        pass

    @abstractmethod
    def grad(self, data, **kwargs):
        # calculate grad_x h(x)
        pass  

    @abstractmethod
    def jacobian(self, data, **kwargs):
        # calculate J h(x)
        pass 


@register_operator(name='inpainting')
class InpaintingOperator(Operator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        # data: (..., C, H, W)
        mask = kwargs.get("mask").to(self.device) # (C, H, W) or (G, C, H, W)
        selected_data = th.masked_select(data, mask) # return one dimensional object 
        return selected_data 
    
    def grad(self, data, **kwargs): 
        return th.tensor(1.0, dtype=data.dtype, device=data.device)
    
    def jacobian(self, data, **kwargs):
        return th.diag_embed(th.grad(data, **kwargs))

