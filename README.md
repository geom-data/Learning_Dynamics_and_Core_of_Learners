# Supplementary Code (Anonymous)

This zip contains four notebooks corresponding to the experimental pipeline.

## Notebooks

1. `01_torch_autoattack_robustbench.ipynb`
   - Generate transfer-based AutoAttack adversarial examples using RobustBench pretrained models.

2. `02_tf_train_resnet_vgg.ipynb`
   - Train `.keras` models (ResNet/VGG) and save weights to `data/models/`.

3. `03_tf_art_apgd_samples.ipynb`
   - Generate APGD samples using ART in TF2 against $\mathcal{U}^{(0)}$ baseline ensembles of four ResNet and four VGG models, and $\mathcal{U}^{(1)}$ the first immunized generation, and save to `data/adversarial_samples/art/`.

4. `04_main_experiment.ipynb`
