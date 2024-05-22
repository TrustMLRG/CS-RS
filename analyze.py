import numpy as np
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math
import pdb
sns.set()

class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()

class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str,type):
        self.data_file_path = data_file_path
        self.type=type
    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):     
        if self.type=='single_seed':
            return ((df['label']==3)&df['correct']&(df['radius']>=radius)).sum()/(df['label']==3).sum()

        elif self.type=='multi_seed':
            return ((df["radius"] >= radius)|((df['label']==2)&(df['predict']==4))).sum()/((df['label']==2)|(df['label']==4)).sum()
        elif self.type=='single_pair':
            return ((df['label']==3)&(df['predict']!=5)&(df['radius']>=radius)).sum()/(df['label']==3).sum() 
        elif self.type=='multi_pair':
            return ((df["predict"]!=2)&(df['predict']!=4)&(df['predict']!=5) & (df["radius"] >= radius)).mean()

        
class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))

class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x

def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    import matplotlib as mpl
    import numpy as np

    # 设置全局字体大小和加粗效果
    # mpl.rcParams['font.size'] = 14
    # mpl.rcParams['font.weight'] = 'bold'
    radii = np.arange(0, max_radius + radius_step, radius_step)
    # plt.figure()
    plt.figure(figsize=(8, 6))
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=20)
    plt.xlabel("radius", fontsize=20)
    plt.ylabel("certified accuracy",fontsize=20)
    plt.legend([method.legend for method in lines], loc='upper right', prop={'size': 20})
    # plt.tight_layout()
    plt.savefig(outfile + ".pdf",dpi=600)
    
    plt.title(title, fontsize=20)
    # plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=600)
    plt.close()

if __name__ == "__main__":

    plot_certified_accuracy(
     dir+"single_seed", "", 2.0, [ 
         Line(ApproximateAccuracy(dir+"motivation/ours_seed3_cer",'single_seed'), "Ours"),
         Line(ApproximateAccuracy(dir+"motivation/salman_seed3_lbd0.1_cer",'single_seed'), "SmoothAdv-R"),
         Line(ApproximateAccuracy(dir+"ddl_cer/cohen_cifar_beta0.1_cer",'single_seed'), "Gaussian-R"),
         Line(ApproximateAccuracy(dir+"motivation/ori_macer_cer",'single_seed'), "MACER"),
         Line(ApproximateAccuracy(dir+"motivation/SmoothMixR_cer",'single_seed'), "SmoothMix-R"),    
     ])

    

