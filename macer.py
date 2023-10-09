'''
Provably Robust Cost-sensitive learning
References:
[1] J. Cohen, E. Rosenfeld and Z. Kolter. 
Certified Adversarial Robustness via Randomized Smoothing. In ICML, 2019.
[2] R. Zhai, C. Dan et al. "MACER: Attack-free and scalable robust training via maximizing certified radius." In ICLR, 2020.
Acknowledgements:
[1] https://github.com/locuslab/smoothing/blob/master/code/certify.py
[2] https://github.com/RuntianZ/macer
'''

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pdb
import numpy as np
from rs.certify import certify
from train_utils import AverageMeter,accuracy

def train(sigma, lbd1,lbd2, gauss_num, beta, gamma, num_classes,
                    model, trainloader, optimizer, device,epoch,gamma1,gamma2,seed_type):
  m = Normal(torch.tensor([0.0]).to(device),
             torch.tensor([1.0]).to(device))

  cl_total = 0.0
  rl_total = 0.0
  rl_sensitive = 0.0
  rl_normal = 0.0
  input_total = 0
  total_correct=[]

  for i, (inputs, targets) in enumerate(trainloader):
    
    inputs, targets = inputs.to(device), targets.to(device)
 
    # breakpoint()
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
    top2_score = top2[0] #(64,2)
    top2_idx = top2[1] #(64,2)

    indices_correct = (top2_idx[:, 0] == targets)  # G_theta
    total_correct += indices_correct.detach().cpu().tolist()
    out0, out1 = top2_score[indices_correct,0], top2_score[indices_correct, 1]

     
    one_hot_labels = torch.eye(num_classes)[targets.cpu()].to('cuda')
    first_posterior = torch.masked_select(beta_outputs_softmax,one_hot_labels.bool())
    second_posterior, _ = torch.max((1-one_hot_labels)*beta_outputs_softmax,dim=1)
    
    if seed_type.isnumeric():
        mask = (targets == int(seed_type)).bool()
    elif seed_type == 'multi':
        mask = ((targets==2)|(targets==4)).bool()

    rob_loss = m.icdf(first_posterior) - m.icdf(second_posterior)
    sensitive_indices = ~torch.isnan(rob_loss) & ~torch.isinf(rob_loss)&(torch.abs(rob_loss) < gamma1)&mask 
     
    sensitive_loss = m.icdf(second_posterior[sensitive_indices])-m.icdf(first_posterior[sensitive_indices])+gamma1

    normal_indices = ~torch.isnan(rob_loss) & ~torch.isinf(rob_loss)&(rob_loss > 0) &(rob_loss<gamma2)&(~mask)
    
    normal_loss = m.icdf(second_posterior[normal_indices])-m.icdf(first_posterior[normal_indices])+gamma2
    # our_rob_loss = sensitive_loss+normal_loss
    # robustness_loss = lbd1*sensitive_loss.sum() + lbd2*normal_loss.sum()
    robustness_loss = sensitive_loss.sum() + normal_loss.sum()
    robustness_loss = robustness_loss.sum() * sigma / 2

    rl_total += robustness_loss.item()
    rl_sensitive += (sensitive_loss.sum()*sigma/2).item()
    rl_normal  += (normal_loss.sum()*sigma/2).item()
    if robustness_loss == 0:
        loss = classification_loss
    else:
        loss = classification_loss + lbd1*robustness_loss
    loss /= input_size

    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)
    if loss != loss:
      raise Exception('NaN in loss, crack!')

    optimizer.step()

  acc = sum(total_correct)/input_total
  cl_total /= input_total
  rl_total /= input_total
  print('Classification Acc: {} Classification Loss: {}  Sensitive Rob Loss: {}  Normal Rob Loss: {}'.format(acc, cl_total, rl_sensitive, rl_normal))
