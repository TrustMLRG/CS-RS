'''
Provably Robust Cost-Sensitive Learning via Randomized Smoothing
'''
import argparse
import numpy as np
import time
import os
import pdb 

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, SVHN
from tqdm import tqdm
from train_utils import AverageMeter, accuracy, init_logfile, log

from model import resnet110, LeNet
from architectures import ARCHITECTURES, get_architecture
from datasets import get_dataset, DATASETS
from certify import certify
from process_HAM10k import load_ham_data

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='MACER Train and Test')
  
  parser.add_argument('--seed', default=1, type=int, metavar='N', help='seed')
  parser.add_argument('--root', default='dataset_cache/', type=str, help='Dataset path')
  parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset:imagenet,cifar10')
  parser.add_argument('--resume_ckpt', default=None, type=str,
                      help = 'Checkpoint path to resume')
  parser.add_argument('--ckptdir', default='ckpt/tmp/', type=str,
                      help = 'Checkpoints save directory')
  parser.add_argument('--matdir', default='mat/tmp/', type=str,
                      help = 'Matfiles save directory')

  parser.add_argument('--epochs', default = 440,
                      type = int, help='Number of training epochs')
  parser.add_argument('--gauss_num', default = 16, type=int,
                      help='Number of Gaussian samples per input')
  parser.add_argument('--batch_size', default=64, type=int, help='Batch size') 

  # params for train
  parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
  parser.add_argument('--sigma', default=0.5, type=float,
                      help='Standard variance of gaussian noise (also used in test)')
  parser.add_argument('--lbd1', default = 3, type=float,
                      help='Weight of robustness loss')
  parser.add_argument('--lbd2', default = 3, type=float,
                      help='Weight of robustness loss')
  parser.add_argument('--gamma', default=8, type=float,
                      help='Hinge factor')
  parser.add_argument('--left', default=-16.0, type=float,
                      help='left value for margin')
  parser.add_argument('--gamma1', default=16.0, type=float,
                      help='Sensitive Hinge factor')
  parser.add_argument('--gamma2', default=4.0, type=float,
                      help='Non Sensitive Hinge factor')
  parser.add_argument('--beta', default=16.0, type=float,
                      help='Inverse temperature of softmax (also used in test)')
  parser.add_argument('--seed_type', default=3, type=str,
                      help='seed type for single seed class') # 3

  # params for test
  parser.add_argument('--skip', default=1, type=int,
                      help = 'Number of skipped images per test image')
  parser.add_argument('--num_classes', default=10, type=int,
                      help = 'Number of classes.')
  parser.add_argument('--version', default='v0', type=str,
                      help = 'version of macer,v0: only correct cat; v1:')
  parser.add_argument('--outfile', default='v1', type=str,
                      help = 'version of macer,v0: only correct cat; v1:')
  parser.add_argument('--arch', default='cifar_resnet56', type=str,
                      help = 'end index for certification')
  parser.add_argument('--target_type', default='single', type=str,
                      help = 'target type in pair-wise case: single or multiple')
  parser.add_argument('--cs', default=True, type=bool,
                      help = 'short for contain sensitive,whether to contain sensitive class in normal radius optimization \
                      for keep sensitive class overall accuracy in pair-wise definition')
# params for certify
  parser.add_argument("--N0", type=int, default=100)
  parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
  parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
  parser.add_argument("--certify_batch", type=float, default=1000, help="batch_size for certification")

  args = parser.parse_args()

  if args.version == 'v0':
    from macer import train
  elif args.version=='v1':
    from pair_macer import train

  ckptdir = None if args.ckptdir == 'none' else args.ckptdir
  matdir = None if args.matdir == 'none' else args.matdir
  if matdir is not None and not os.path.isdir(matdir):
    os.makedirs(matdir)
  if ckptdir is not None and not os.path.isdir(ckptdir):
    os.makedirs(ckptdir)
  checkpoint = None if args.resume_ckpt == 'none' else args.resume_ckpt
  
  # Load dataset and build model
  if args.dataset == 'mnist':
     
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    base_model = LeNet()
    trainset = MNIST(
        root = args.root, train=True, download=True, transform=transform_train)
    testset = MNIST(
        root = args.root, train=False, download=True, transform=transform_test)

  elif args.dataset == 'cifar10':

    
    base_model = get_architecture(args.arch,'cifar10')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = CIFAR10(
        root = args.root, train=True, download=True, transform=transform_train)
    testset = CIFAR10(
        root = args.root, train=False, download=True, transform=transform_test)

  elif args.dataset == 'ham':
    base_model = get_architecture('cifar_resnet56','ham')
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
            [
                transforms.Resize(299), #299
                transforms.CenterCrop(299), #299
                transforms.ToTensor(),
                normalize
            ]
    )
    trainset,testset = load_ham_data(transform)
   
  pin_memory = (args.dataset=='imagenette')
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=args.batch_size, shuffle = True, pin_memory=pin_memory,num_workers=1)

  num_classes = args.num_classes

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cuda':
    cudnn.benchmark = True
    model = torch.nn.DataParallel(base_model)

  optimizer = optim.SGD(
      model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
  scheduler = MultiStepLR(
      optimizer, milestones = [200,400], gamma=0.1) # [200, 400]

  # Resume from checkpoint if required

  start_epoch=0
  if checkpoint is not None:
    print('==> Resuming from checkpoint..')
    print(checkpoint)
    checkpoint = torch.load(checkpoint)
    base_model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    scheduler.step(start_epoch)

  for epoch in tqdm(range(start_epoch + 1, args.epochs + 1)):
    print('===train(epoch={})==='.format(epoch))
    t1 = time.time()
    model.train()  

    train(args.sigma, args.lbd1,args.lbd2, args.gauss_num, args.beta,
                    args.target_type, num_classes, model, trainloader, optimizer, device,epoch,args.gamma1,args.gamma2,args.seed_type)
    # epoch -> args.cs
    
    scheduler.step()
    t2 = time.time()
    print('Elapsed time: {}'.format(t2 - t1))

    if ckptdir is not None and epoch%20==0:
        # Save checkpoint
        print('==> Saving {}.pth..'.format(epoch))
        try:
          state = {
              'net': base_model.state_dict(),
              'epoch': epoch,
          }
          torch.save(state, '{}/{}.pth'.format(ckptdir, epoch))
        except OSError:
          print('OSError while saving {}.pth'.format(epoch))
          print('Ignoring...')
