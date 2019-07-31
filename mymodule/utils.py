import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import copy

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def none_string(s):
    if s.lower() == 'none':
        return None
    return s.lower()

def teach_class(logit_stu, logit_tea, class_name, _lambda):
    if class_name is None:
        return 0
    logit_stu = logit_stu.view(-1, logit_stu.size(-1)).float()
    logit_tea = logit_tea.view(-1, logit_tea.size(-1)).float()
    if class_name == "prob":
        logprob_stu = F.log_softmax(logit_stu, 1)
        logprob_tea = F.log_softmax(logit_tea, 1)
        return F.mse_loss(logprob_tea.exp(),logprob_stu.exp())*_lambda
    elif class_name == "logit":
        return F.mse_loss(logit_stu,logit_tea)*_lambda
    elif class_name == "smart":
        prob_stu = F.log_softmax(logit_stu, 1).exp()
        prob_tea = F.log_softmax(logit_tea, 1).exp()
        r_stu = -(1/(prob_stu+1e-6)-1+1e-6).detach().log()
        r_tea = -(1/(prob_tea+1e-6)-1+1e-6).detach().log()
        return (prob_stu*(r_stu-r_tea)*2).mean()*_lambda
        # return (prob_stu*(r_stu-r_tea)*2).mean()
    elif class_name == 'kl':
        logprob_stu = F.log_softmax(logit_stu, 1)
        prob_tea = F.log_softmax(logit_tea, 1).exp()
        return -(prob_tea*logprob_stu).sum(-1).mean()*_lambda
    elif class_name == 'distill':
        temp = 2
        logprob_stu = F.log_softmax(logit_stu/temp, 1)
        prob_tea = F.log_softmax(logit_tea/temp, 1).exp()
        return -(prob_tea*logprob_stu).sum(-1).mean()*_lambda

def create_noisycopy_model(model,args):
    noisycopy_model = copy.deepcopy(model)
    for name,_ in model.named_parameters():
        rec_delete_param(noisycopy_model,name)
    noisycopy_model.noisycopy_eps = args.noisycopy_eps
    noisycopy_model.advcopy_eps = args.advcopy_eps
    return noisycopy_model

def update_noisycopy_model(noisycopy_model,model):
    for name,param in model.named_parameters():
        param_new = param + param.data.new_zeros(param.shape).normal_(0, noisycopy_model.noisycopy_eps).detach()
        rec_setattr(noisycopy_model,name,param_new)
    return noisycopy_model

def update_advcopy_model(noisycopy_model,model):
    # rescale grad
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.float().norm(2)
        if (torch.isnan(param_norm) or torch.isinf(param_norm)):
            return None, True
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    _eps = noisycopy_model.advcopy_eps / (total_norm + 1e-6)

    for name,param in model.named_parameters():
        param_new = param + param.grad.data.detach()*_eps
        rec_setattr(noisycopy_model,name,param_new)
    return noisycopy_model, False

def rec_register_param(obj, attr, value):
    if '.' not in attr:
        obj.register_parameter(attr, value)
    else:
        L = attr.split('.')
        return rec_register_param(getattr(obj, L[0]), '.'.join(L[1:]), value)

def rec_delete_param(obj, attr):
    if '.' not in attr:
        del obj._parameters[attr]
    else:
        L = attr.split('.')
        return rec_delete_param(getattr(obj, L[0]), '.'.join(L[1:]))

def rec_getattr(obj, attr):
    if '.' not in attr:
        return getattr(obj, attr)
    else:
        L = attr.split('.')
        return rec_getattr(getattr(obj, L[0]), '.'.join(L[1:]))

def rec_setattr(obj, attr, value):
    if '.' not in attr:
        obj.register_buffer(attr, value)
    else:
        L = attr.split('.')
        rec_setattr(getattr(obj, L[0]), '.'.join(L[1:]), value)
