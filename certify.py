# evaluate a smoothed classifier on a dataset
import argparse
import os
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes

from time import time
import torch
import datetime
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from architectures import get_architecture
import pdb
import os

from matplotlib.ticker import FuncFormatter
plt.switch_backend('agg')
plt.rcdefaults()
from tqdm import tqdm

from core import Smooth

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file for the second radius")
parser.add_argument("--batch", type=int, default=400, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--low", type=int, default=0, help="number of samples to use")
parser.add_argument("--high", type=int, default=10000, help="number of samples to use")
parser.add_argument("--type", type=str, default='macer', help="cohen or macer mnist style ckpt")
parser.add_argument("--target_type", type=str, default='single', help="single target or multiple target")
parser.add_argument("--certify", type=str, default='overall', help="certify sensitive or overall")
parser.add_argument("--seed_value", type=int, default=3, help="seed value") 

args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    
    if args.type=='cohen':
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])
    elif args.type=='macer':
        base_classifier = get_architecture('cifar_resnet56', args.dataset)
        base_classifier.load_state_dict(checkpoint['net'])
    else:
        raise ValueError  

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f1 = open(args.outfile,'a')

    print("idx\tlabel\tpredict\tr\tr1\tr2\tmass\tnA\tnB\tcBHat\tcorrect", file = f1, flush = True)
    dic = {}
    counts_dic = {}
    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    flag_list = []
    for i in tqdm(range(args.low, len(dataset))): # max:
        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        (x, label) = dataset[i]
        x = x.cuda()

        before_time = time()
        if args.certify == 'sensitive':
            if label!=args.seed_value:
                continue
            else:
                if args.target_type == 'single':
                    if args.dataset == 'cifar10':
                        target_list = [5] #3->5
                    elif args.dataset == 'imagenette':
                        target_list = [2] #7->2

                elif args.target_type == 'multi':
                    if args.dataset == 'cifar10':
                        target_list=[2,4,5]
                    elif args.dataset == 'imagenette':
                        target_list = [2,4,6]  
                elif args.target_type=='multi2':
                    if args.dataset == 'cifar10':
                        target_list=[2,4,5,7,9]
                    else:
                        pass
                elif args.target_type == 'multi3':
                    target_list = [i for i in range(10) if i != label]
                prediction, r, r1, r2, mass, nA,nB,cBHat  = smoothed_classifier.certify(i,x, args.N0, args.N, args.alpha, args.batch,label,target_list)
                correct = int (prediction not in target_list)

        else:
            target_list = [i for i in range(10) if i != label]
           
            prediction, r, r1, r2, mass, nA,nB,cBHat  = smoothed_classifier.certify(i,x, args.N0, args.N, args.alpha, args.batch,label,target_list)
            correct = int (prediction not in target_list)

        after_time = time() 
        time_elapsed = str(datetime.timedelta(seconds = (after_time - before_time)))
        final_str = "{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{}\t{}\t{}\t{}".format(i, label, prediction,r,r1,r2,mass,nA,nB,cBHat,correct)

        print(final_str, file = f1, flush = True)





