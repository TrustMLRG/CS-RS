# Provably Robust Cost-Sensitive Learning via Randomized Smoothing
  The goal of this project:
  - We study whether randomized smoothing, a scalable certification framework, can be leveraged to certify cost-sensitive robustness.
  - Built upon a notion of cost-sensitive certified radius, we show how to adapt the standard randomized smoothing certification pipeline to produce tight robustness guarantees for any given cost matrix.
  - with fine-grained certified radius optimization schemes designed for different data subgroups, we propose an algorithm to train smoothed classifiers that are optimized for cost-sensitive robustness

# Certification for cost-sensitive robustness

` python certify.py --dataset cifar10 --base_classifier checkpoint_dir --sigma 0.5 --outfile outfile --certify sensitive --type macer`


# Training for seedwise cost matrices
` python train.py --dataset cifar10 --version v0 --ckptdir ckpt --lbd 3 --outfile outfile --arch cifar_resnet56 --sigma 0.5`

# Training for pairwise cost matrices
` python train.py --dataset cifar10 --version v1 --ckptdir ckpt --lbd 3 --outfile outfile --arch cifar_resnet56 --sigma 0.5`


