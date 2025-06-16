# Training Samples from CIFAR10 Dataset


![Original CIFAR10 training dataset images](/images/originalCIFAR10.png)

### Fast Gradient Method

![Adversarial Images through FGM](/images/AdversarialAttackusingFGM.png)

### Projected Gradient Descent

1. BIM and Madry's method are basically PGD with slightly different options.
2. PGD is iterative method of FGM

### CWL2

This attack is an iterative attack using Adam and a specially-chosen loss function to find adversarial examples with ***lower distortion*** than other attacks. 

This comes at the cost of speed, as this attack is often ***much slower*** than others.


### Momentum Iterative Method

Normalize current gradient and add it to the accumulated gradient

momentum = decay_factor * momentum + grad

Itertative method with mometum.

### Simultaneous Perturbation Stochastic Approximation (SPSA)

Optimizer for gradient-free attacks in https://arxiv.org/abs/1802.05666.

Gradients estimates are computed using Simultaneous Perturbation Stochastic Approximation (SPSA),
combined with the ADAM update rule (https://arxiv.org/abs/1412.6980).