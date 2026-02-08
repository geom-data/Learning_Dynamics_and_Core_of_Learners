## Notebooks (recommended order)

### `notebooks/01_torch_autoattack_robustbench.ipynb`

Generates transfer-based **AutoAttack (L2)** adversarial examples using **RobustBench** pretrained models (PyTorch), and saves outputs under:
 `data/adversarial_samples/autoattack_torch/`

### `notebooks/02_tf_train_resnet_vgg.ipynb`

Trains CIFAR-10 classifiers in **TF2/Keras** and saves `.keras` checkpoints under:
`data/models/` (=`tf_model_dir` in `configs/paths.yaml`)

Models trained:

ResNet: `resnet_v1_n3_d20`, `resnet_v1_n9_d56`, `resnet_v2_n3_d20`, `resnet_v2_n9_d56`

VGG: `vgg11_raw`, `vgg13_raw`, `vgg16_raw`, `vgg19_raw`

### `notebooks/03_tf_art_apgd_samples.ipynb`

Generates **APGD** adversarial samples via **ART (TF2)** against:

baseline ensemble $\mathcal{U}^{(0)}$ (4 ResNet + 4 VGG)

first immunized generation $\mathcal{U}^{(1)}$ ( $\mathcal{U}^{(0)}$ + 4 specialized models on weak-APGD against $\mathcal{U}^{(0)}$ )

Outputs:
`data/adversarial_samples/ART/`

### `notebooks/04_main_experiment.ipynb`

Runs the main evaluation pipeline:

1. loads models + adversarial samples

2. builds the immunization mechanism / specialization

3. produces metrics, figures, and logs under `results/figures/` and `results/logs/`.

### Notes

- Paths are configured in `configs/paths.yaml`; experiment settings (e.g., seed) in `configs/exp.yaml`.
- If you downloaded models from GitHub Releases, extract:

  `models.zip` → `data/models/`

  `specialized_models.zip` → `data/specialized_models/`
- If you downloaded adversarial samples from GitHub Releases, extract:

  `adversarial_samples.zip` → `data/adversarial_samples/`
  