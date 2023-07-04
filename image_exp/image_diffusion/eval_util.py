import numpy as np 
from scipy.special import logsumexp
import torch 


def pred(resnet_model, data, return_probs=True):
    # resnet_model already takes care of scaling data from range [-1,1] to [0,1]
    data_t = torch.tensor(data, device=torch.device("cuda:0"))
    logits = resnet_model(data_t) # (P, num_classes)
    if return_probs:
        pred_probs = torch.softmax(logits, dim=-1) # (P, num_classes) 
        return logits.detach().cpu().numpy(), pred_probs.detach().cpu().numpy()   
    return logits.detach().cpu().numpy()


def compute_weighted_accuracy(resnet_model, data, log_w, label, return_logits=False):
    if isinstance(log_w, list):        
        weights_list = [normalize_weights_from_log_weights_np(log_w_) for log_w_ in log_w]
    else: 
        weights_list = [normalize_weights_from_log_weights_np(log_w)]

    logits, pred_probs = pred(resnet_model, data, return_probs=True)  

    # compute two type of accuracy -- one averaged over prediction 
    # the other one averaged over probs 
    
    pred_classes_by_particles = np.argmax(logits, axis=-1) # (P,)
    
    pred_accuracy_list = []
    bayes_accuracy_list = [] 

    for weights in weights_list:
        
        pred_accuracy = ((pred_classes_by_particles == label) * weights).sum()
        pred_accuracy_list.append(pred_accuracy)
        
        weighted_pred_probs = np.sum(pred_probs * weights[:, None] , axis=0) # (num_classes, )
        prediction = np.argmax(weighted_pred_probs, axis=-1) # scalar 
        bayes_accuracy = (prediction == label) # scalar
        bayes_accuracy_list.append(bayes_accuracy)
    
    if len(pred_accuracy_list) == 1:
        assert len(bayes_accuracy_list) == 1, len(bayes_accuracy_list)
        if return_logits:
            return pred_accuracy_list[0], bayes_accuracy_list[0], logits 
        return pred_accuracy_list[0], bayes_accuracy_list[0]
    
    if return_logits:
        return pred_accuracy_list, bayes_accuracy_list, logits 
    return pred_accuracy_list, bayes_accuracy_list 



def compute_weighted_accuracy_from_logits_np(logits, log_w, label, return_logits=False):
    if isinstance(log_w, list):        
        weights_list = [normalize_weights_from_log_weights_np(log_w_) for log_w_ in log_w]
    else: 
        weights_list = [normalize_weights_from_log_weights_np(log_w)]
    
    # logits (P, num_classes)
    logits = logits - logits.max(axis=-1, keepdims=True) 
    pred_probs = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True)) # (P, num_classes)

    # compute two type of accuracy -- one averaged over prediction 
    # the other one averaged over probs 
    
    pred_classes_by_particles = np.argmax(logits, axis=-1) # (P,)
    
    pred_accuracy_list = []
    bayes_accuracy_list = [] 
    
    for weights in weights_list:
        
        pred_accuracy = ((pred_classes_by_particles == label) * weights).sum()
        pred_accuracy_list.append(pred_accuracy)
        
        weighted_pred_probs = np.sum(pred_probs * weights[:, None] , axis=0) # (num_classes, )
        prediction = np.argmax(weighted_pred_probs, axis=-1) # scalar 
        bayes_accuracy = (prediction == label) # scalar
        bayes_accuracy_list.append(bayes_accuracy)
    
    if len(pred_accuracy_list) == 1:
        assert len(bayes_accuracy_list) == 1, len(bayes_accuracy_list)
        if return_logits:
            return pred_accuracy_list[0], bayes_accuracy_list[0], logits 
        return pred_accuracy_list[0], bayes_accuracy_list[0]
    
    if return_logits:
        return pred_accuracy_list, bayes_accuracy_list, logits 
    return pred_accuracy_list, bayes_accuracy_list 


def normalize_weights_from_log_weights_np(log_w):
    log_w = log_w - log_w.max()
    weights = np.exp(log_w - logsumexp(log_w))
    return weights 


def find_value(s, str_name):
    for subs in s.split("_"):
        if str_name in subs:
            return subs[len(str_name):] # string 
            
            