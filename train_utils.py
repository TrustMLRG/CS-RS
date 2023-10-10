import torch
import pdb
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        
        batch_size = target.size(0)
        mask=target==3
        cat_output=output[mask]
        cat_target=target[mask]
        bird_mask=target==2
        bird_output=output[bird_mask]
        bird_target = target[bird_mask]
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)) #(5,batch_size)
        
        _, cat_pred = cat_output.topk(maxk, 1, True, True)
        cat_pred = cat_pred.t()
        cat_correct = cat_pred.eq(cat_target.view(1, -1).expand_as(cat_pred)) #(5,batch_size)
        
        _, bird_pred = bird_output.topk(maxk, 1, True, True)
        bird_pred = bird_pred.t()
        bird_correct = bird_pred.eq(bird_target.view(1, -1).expand_as(bird_pred)) #(5,batch_size)

        res = []
        cat_res,bird_res = [],[]
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            
        return res[0],res[1],

def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()
