'''
Single Target
Multi Target
'''
from logging import raiseExceptions
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pdb
import numpy as np
from rs.certify import certify
from train_utils import AverageMeter,accuracy

def train(sigma, lbd,lbd2, gauss_num, beta, target_type, num_classes,
                    model, trainloader, optimizer, device,cs,gamma1,gamma2,seed_type):

  m = Normal(torch.tensor([0.0]).to(device),
             torch.tensor([1.0]).to(device))

  cl_total = 0.0
  rl_total = 0.0
  input_total = 0
  total_correct=[]
  for i, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    input_size = len(inputs)
    input_total += input_size

    new_shape = [input_size * gauss_num]
    new_shape.extend(inputs[0].shape)
    inputs = inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
    noise = torch.randn_like(inputs, device=device) * sigma
    noisy_inputs = inputs + noise

    outputs = model(noisy_inputs)
    outputs = outputs.reshape((input_size, gauss_num, num_classes))
    
    # Classification loss
    outputs_softmax = F.softmax(outputs, dim=2).mean(1)
    outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
    classification_loss = F.nll_loss(
        outputs_logsoftmax, targets, reduction='sum')
    cl_total += classification_loss.item()

    # Robustness loss
    beta_outputs = outputs * beta  # only apply beta to the robustness loss
    beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1) #(64,10)
    top2 = torch.topk(beta_outputs_softmax, 2) #(64,10)
    top2_idx = top2[1] #(64,2)

    indices_correct = (top2_idx[:, 0] == targets)  # G_theta
    total_correct += indices_correct.detach().cpu().tolist()
    one_hot_labels = torch.eye(num_classes)[targets.cpu()].to('cuda')
    first_posterior = torch.masked_select(beta_outputs_softmax,one_hot_labels.byte())
    second_posterior, _ = torch.max((1-one_hot_labels)*beta_outputs_softmax,dim=1)
    # multi_target 
    if target_type == 'multi': 
        mis_targets = torch.tensor([2,4,5]) # for mnist (0,2,6,8,9)
    elif target_type == 'single':
        mis_targets = torch.tensor([5]) # 5 for cifar ; 0,2 for ham
    else:
        raise ValueError(target_type)
    num = len(mis_targets)
    mis_targets = mis_targets.unsqueeze(0)
    mis_onehot = torch.zeros(mis_targets.size(0), num_classes).scatter_(1, mis_targets, 1.)
    mis_onehot = mis_onehot.repeat(input_size,1).cuda()
    
    seed_posterior,_ = torch.max(beta_outputs_softmax,dim=1) # j 
    target_posterior, _ = torch.max((mis_onehot)*beta_outputs_softmax,dim=1)
    
    # actual radius
    overall_radius =  m.icdf(first_posterior) - m.icdf(second_posterior)
    sensitive_radius = m.icdf(seed_posterior) - m.icdf(target_posterior)
    
    mask = (targets == int(seed_type)).bool() # (3 -> 2,4,5) # 3 for cifar, 4 for mnist

    sensitive_indices = ~torch.isnan(sensitive_radius) & ~torch.isinf(sensitive_radius)&(torch.abs(sensitive_radius) < gamma1)&mask 
    # sensitive_indices = ~torch.isnan(sensitive_radius) & ~torch.isinf(sensitive_radius)&(sensitive_radius > 0)&(sensitive_radius < gamma1)&mask 
    sensitive_loss = m.icdf(target_posterior[sensitive_indices])-m.icdf(seed_posterior[sensitive_indices])+gamma1

    if cs : 
        normal_indices = ~torch.isnan(overall_radius) & ~torch.isinf(overall_radius)&(overall_radius > 0) &(overall_radius<gamma2)
    else:
        normal_indices = ~torch.isnan(overall_radius) & ~torch.isinf(overall_radius)&(overall_radius > 0) &(overall_radius<gamma2)&(~mask)
    
    normal_loss = m.icdf(second_posterior[normal_indices])-m.icdf(first_posterior[normal_indices])+gamma2
    robustness_loss = (sensitive_loss.sum() + normal_loss.sum()) * sigma / 2
    # robustness_loss = normal_loss.sum() * sigma / 2 # only normal loss should be equal to macer

    rl_total += robustness_loss.item()
    loss = classification_loss + lbd * robustness_loss
    loss /= input_size
    optimizer.zero_grad()
    loss.backward()
    # only used for mnist
    # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)
    if loss != loss:
      raise Exception('NaN in loss, crack!')
    optimizer.step()

  acc=sum(total_correct)/input_total
  cl_total /= input_total
  rl_total /= input_total
  print('Classification Acc: {} Classification Loss: {}  Robustness Loss: {}'.format(acc, cl_total, rl_total))
