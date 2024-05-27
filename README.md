# Provably Robust Cost-Sensitive Learning via Randomized Smoothing
  The goal of this project:
  - We study whether randomized smoothing, a scalable certification framework, can be leveraged to certify and train for cost-sensitive robustness.
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
      
  - Certification for overall robustness
    ```text
    python certify.py --dataset cifar10 --base_classifier checkpoint_dir --sigma 0.5 --outfile outfile --certify overall --type macer
    ```
    
  - Certification for cost-sensitive robustness
    ```text
    python certify.py --dataset cifar10 --base_classifier checkpoint_dir --sigma 0.5 --outfile outfile --certify sensitive --type macer
    ```


  - Training for seedwise cost matrices
    ```text
    python train.py --dataset cifar10 --version v0 --ckptdir ckpt --lbd1 3  --sigma 0.5 --seed_type 3
    ```

  - Training for pairwise cost matrices
    ```text
    python train.py --dataset cifar10 --version v1 --ckptdir ckpt --lbd1 3  --sigma 0.5 --seed_type 3 --target_type single
    ```

# What is in this repository?
* ```CS-RS```, including:
  * ```train.py```: implements detailed training pipeline
  * ```macer.py, pair_macer.py```:  main functions for training provably robust cost-sensitive classifiers for seedwise and pairwise cost matrices
  * ```certify```: implements detailed certification pipeline 
  * ```core.py```: implements detailed practical certification algorithm for cost-sensitive robustness
  * ```train_gaussian_R.py, train_salman_R.py,train_smoothmix_R.py```: implements the reweighting adaptation for three baselines: mstandard gaussian, smoothadv and smoothmix
  * ```analyze.py```: functions to draw the cost-sensitive robustness v.s. certified radius curves
 
