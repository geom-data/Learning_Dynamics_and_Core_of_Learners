# Learning Dynamics and Core of Learners experiments

This repository contains the experimental part for the paper:

**Learning Dynamics and the Core of Learners**  
Paper (arXiv): https://arxiv.org/abs/2602.05026

We conduct experiment to (i) compare entropy-based uncertainty of ensemble of multiple classifiers and single models, (ii) evaluate memory formation accross environments under the proposed learning dynamics for ensembles with implementation of a two-generation logifold (IMM: Immunization Mechanism), and (iii) study IMM behaviour against transfer-based adversarial attacks in terms of core accuracy and core coverage.

## Key idea

Use ensemble entropy (disagreement / predictive uncertainty) to split inputs into:

- core (low entropy, reliable) -> handled by the first generation
- out-of-core (high entropy, suspicious / shifted) -> routed to specialists specialized(fine-tuned) on high-entropy samples

This yields large gains under perturbations (see the paper tables reproduced by 04_main_experiment.ipynb) while pertaining memory on original samples.

## Notebooks

This project mixes PyTorch (RobustBench/AutoAttack) and TensorFlow 2 (training + ART APGD + main pipeline).
See env/requirements-*.txt

### `01_torch_autoattack_robustbench.ipynb`(Pytorch)

Generates transfer-based [AutoAttack](https://github.com/fra31/auto-attack) (L2) adversarial examples using [RobustBench](https://robustbench.github.io/) pretrained models (PyTorch), and saves outputs under:
 `data/adversarial_samples/autoattack_torch/`

Threat model is untargeted L2 epsilon = 0.5 *standard* surrogate from RobustBench.

### `02_tf_train_resnet_vgg.ipynb`(TensorFlow 2)

Trains CIFAR-10 classifiers in **TF2/Keras** and saves `.keras` checkpoints under:
`data/models/` (=`tf_model_dir` in `configs/paths.yaml`)

Models trained:

ResNet: `resnet_v1_n3_d20`, `resnet_v1_n9_d56`, `resnet_v2_n3_d20`, `resnet_v2_n9_d56`
VGG: `vgg11_raw`, `vgg13_raw`, `vgg16_raw`, `vgg19_raw`

Implementation references:

- ResNet: [`source/resnet.py`](source/resnet.py) [reference](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | Based on Keras example cifar10_resnet.py
- VGG: [`source/vgg.py`](source/vgg.py) [reference](https://ieeexplore.ieee.org/document/7486599) | The VGG implementation follows the standard torchvision VGG configuration, and the training utilities are adapted from the Keras CIFAR-10 ResNet example style.

### `03_tf_art_apgd_samples.ipynb`(TensorFlow 2)

Generates **APGD** adversarial samples via [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) (TF2) against:

baseline ensemble $\mathcal{U}^{(0)}$ (4 ResNet + 4 VGG) trained on original CIFAR10 dataset from scratch

first immunized generation $\mathcal{U}^{(1)}$ ( $\mathcal{U}^{(0)}$ + 4 specialized models on weak-APGD against $\mathcal{U}^{(0)}$ )

Outputs:
`data/adversarial_samples/apgdce_ART/`

### `04_main_experiment.ipynb`(TensorFlow 2)

Runs the main evaluation pipeline:

loads models + adversarial samples

**Experiment 1**  

Compute ensemble and single model entropies on each white-box attack against themselves with weak/strong configuration. Then evaluate ensemble and single model performance on each white-box attack.

**Experiment 2**

Construct IMM using $\mathcal{U}^{(1)}$ and $\mathcal{U}^{(2)\prime}$, where $\mathcal{U}^{(2)\prime}$ is specialized on high-entropy samples from $\mathcal{U}^{(1)}$ across environments:

1. Select an entropy threshold using validation samples  
2. Train specialists $\mathcal{U}^{(2)\prime}$ on high-entropy samples  
Then evaluate **overall accuracy**, **core accuracy**, and **core coverage**.

**Experiment 3**  

See how core coverage and total entropy behaves

## Notes

### Releases / large files

If you downloaded artifacts from a release, extract them into:

- models.zip -> data/models/
- specialized_models.zip -> data/specialized_models/
- adversarial_samples.zip -> data/adversarial_samples/

### Reproducibility

- Random seed: `configs/exp.yaml` (`random_seed = 42`)
- Paths: `configs/paths.yaml`

Compute environment

The paper experiments were run on the Boston University SCC cluster with an NVIDIA L40S GPU.

## License

This project is released under the MIT License. See `LICENSE` for details.
