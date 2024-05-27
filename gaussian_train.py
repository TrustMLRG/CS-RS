# this file is adapted from https://github.com/locuslab/smoothing
import argparse
import os
from re import A
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import pdb
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log
from tqdm import tqdm

print(torch.cuda.is_available())
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices = DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help = 'folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.5, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--beta', default = 0.1, type = float,
                    help='coefficient factor for original loss and cost sensitive loss')
parser.add_argument('--type', default = 'single', type = str,
                    help = 'cost matrix type for single seed or multi seed')
parser.add_argument('--outfile', default='', type=str,
                      help = 'output certification file for cohen')
parser.add_argument('--skip', default=1, type=int,
                      help = 'Number of skipped images per test image')
args = parser.parse_args()

def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')

    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)
            
    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\tfactor_a\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc\tcat loss\tcat acc\ttest cat loss\ttest cat acc")

    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    train_cat_loss_list = []
    test_cat_loss_list = []
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd,args.beta,args.type)
        test_loss, test_acc = test(test_loader, model, criterion, args.noise_sd) 
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch,scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))
        if (epoch+1)%10==0:
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint{}.pth.tar'.format(epoch+1)))
        
        torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

    model.eval()
    certify(args.outfile, model, device, test_dataset, num_classes=10, skip = args.skip,sigma=0.5,batch=1000,N0=100,N=100000,alpha=0.001)

    return train_cat_loss_list, test_cat_loss_list
    
def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float,beta:float,type:str):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    cat_top1 = AverageMeter()
    cat_top5 = AverageMeter()
    cat_losses = AverageMeter()
    end = time.time()
    cat_loss = []
    # switch to train mode
    model.train()
    
    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda() # (batch_size,3,32,32)
        targets = targets.cuda() 
        
        N = inputs.shape[0] # batch_size
        noise = torch.randn_like(inputs, device = 'cuda') * noise_sd

        if args.dataset=='imagenette':
            if type=='single':
                mask = (targets == 3).int()
            elif type=='multi':
                mask = ((targets==3)|(targets==7)).int()
        elif args.dataset=='cifar10':
            if type=='single':
                mask = (targets == 3).int()
            elif type=='multi':
                mask = ((targets==2)|(targets==4)).int()
        inputs = inputs + noise
        outputs = model(inputs)
        cat_outputs = outputs[mask.bool()]
        N = cat_outputs.shape[0]
        
        cat_targets = targets[mask.bool()]
        loss_cost = criterion(cat_outputs,cat_targets)
        loss_orig = criterion(outputs,targets) # norml loss
        loss = loss_orig + beta*loss_cost
        acc1, acc5= accuracy(outputs, targets, topk=(1, 2))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg) #cat_losses.avg,cat_top1.avg


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cat_top1 = AverageMeter()
    cat_top5 = AverageMeter()
    cat_losses = AverageMeter()
    bird_top1 = AverageMeter()
    bird_top5 = AverageMeter()
    
    end = time.time()

    # switch to eval mode
    model.eval()
     
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()
            mask0 = (targets==3).int()
            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            cat_out = outputs[mask0!=0]
            cat_target = targets[mask0!=0]
            loss_cat = criterion(cat_out,cat_target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 2))  
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
                # print("cat accuracy@1 is:{}, bird accuracy@1 is:{}".format(cat_acc1.item(),bird_acc1.item()))
        return (losses.avg, top1.avg) #, cat_losses.avg, cat_top1.avg


if __name__ == "__main__":
    train_cat_loss, test_cat_loss = main()
