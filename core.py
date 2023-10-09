import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import pdb
import math

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, idx:int,x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int,label:int,sensitive_target:list) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        # pdb.set_trace()
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection, js_list, _ = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class: majority vote
        top2 = counts_selection.argsort()[::-1][:2]

        cAHat = counts_selection.argmax().item() # predicted label
        # draw more samples of f(x + epsilon)
        counts_estimation, _ , _  = self._sample_noise(x, n, batch_size)
        cBHat = sensitive_target[0]
        tmp_n = counts_estimation[cBHat]
        card = len(sensitive_target)
        # for j in sensitive_target:
        #     n_j = counts_estimation[j]
        #     if tmp_n<n_j:
        #         cBHat = j
        #         tmp_n = n_j
        result = [counts_estimation[i] for i in sensitive_target]
        alpha_card_half = alpha / (2 * card)
        pB = self._upper_confidence_bound(result[0],n,alpha_card_half)

        for i,(nb, index) in enumerate(zip(result,sensitive_target)):
            pB_tmp = self._upper_confidence_bound(nb,n,alpha_card_half)
            if pB_tmp > pB:
                cBHat = index
                pB = pB_tmp

        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        
        nB = counts_estimation[cBHat].item()
        mass = nB/(n-nA) if nA<n else 0
        pABar = self._lower_confidence_bound(nA, n, alpha)

        pA = self._lower_confidence_bound(nA,n,alpha/2)
        pB = self._upper_confidence_bound(nB,n,alpha/2*card)

        r1 = self.sigma * norm.ppf(pABar) # standard radius
        r2 = self.sigma * (norm.ppf(pA)-norm.ppf(pB))*0.5 # our calibrated radius
        r = max(r1,r2)
        if pABar < 0.5: 
            return Smooth.ABSTAIN, r, r1, r2, mass,nA,nB,cBHat 
        else:
            return cAHat, r, r1, r2, mass,nA,nB,cBHat

    def predict(self, index, x: torch.tensor, n: int, alpha: float, batch_size: int,label) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts, js_list, predictions = self._sample_noise(x, n, batch_size)

        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        # return top2[0], counts
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN,count1,True
        else:
            return top2[0], count1, False

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            js_list = []
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                # pdb.set_trace()
                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device = 'cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
               
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts, js_list, predictions.cpu().tolist()


    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        
        for idx in arr:
            counts[idx] += 1
            
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha = 2 * alpha, method="beta")[0]

    def _upper_confidence_bound(self, NB: int, N: int, alpha: float) -> float:
        return proportion_confint(NB, N, alpha = 2 * alpha, method="beta")[1]
    
    