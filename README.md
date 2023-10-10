# Provably Robust Cost-Sensitive Learning via Randomized Smoothing
  The goal of this project:
  - We study whether randomized smoothing, a scalable certification framework, can be leveraged to certify cost-sensitive robustness.
  - To certify cost-sensitive robustness, we propose a new practical certification pipeline to produce tight robustness guarantees for any given cost matrix.
  - To train smoothed classifiers that are optimized for cost-sensitive robustness, we propose fine-grained certified radius optimization schemes.

# Installation & Usage
  - Install PyTorch
    ```text
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    ```
   - Install Other Dependencies
      ```text
     pip install scipy pandas statsmodels matplotlib seaborn
      ```
    
  - Certification for cost-sensitive robustness
    ```text
    python certify.py --dataset cifar10 --base_classifier checkpoint_dir --sigma 0.5 --outfile outfile --certify sensitive --type macer
    ```


  - Training for seedwise cost matrices
    ```text
    python train.py --dataset cifar10 --version v0 --ckptdir ckpt --lbd 3 --outfile outfile --arch cifar_resnet56 --sigma 0.5
    ```

  - Training for pairwise cost matrices
    ```text
    python train.py --dataset cifar10 --version v1 --ckptdir ckpt --lbd 3 --outfile outfile --arch cifar_resnet56 --sigma 0.5
    ```

# What is in this respository?
* ```CS-RS```, including:
  * ```train.py```: implements the detailed training and evaluation functions for different classifiers
  * ```macer.py, pair_macer.py```:  main functions for training a provably cost-sensitive robust classifier for seedwise and pairwise cost matrices
  * ```certify```: implements detailed certification pipeline 
  * ```core.py```: detailed practical certification algorithm for cost-sensitive robustness
  * ```smooth_train.py```: implements the reweighting adaptation for standard randomized smoothing training methods
  * ```analyze.py```: functions to draw the cost-sensitive robustness v.s. certified radius curves
 
