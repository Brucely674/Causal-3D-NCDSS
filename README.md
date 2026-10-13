## Motivation

Conventional 3D novel class discovery methods largely rely on statistical correlations. As a result, they may learn **shortcut features** rather than intrinsic class representations, leading to confusion between base and novel categories. We address this issue from a causal perspective: suppressing confounding factors in base-class representations and transferring useful causal knowledge to novel classes.

<p align="center">
  <img src="https://github.com/user-attachments/assets/921bded0-8fda-4b17-a1ef-7c62c25b6e6f" alt="Motivation and structural causal model" width="95%" />
</p>

<p align="center"><em>Motivation: from spurious statistical correlations to causal representations and base-to-novel reasoning.</em></p>

## Framework

<p align="center">
  <img src="https://github.com/user-attachments/assets/53759c65-dbd8-40f2-9e02-2799d561a315" alt="Overview of the Causal-3D-NCDSS framework" width="95%" />
</p>

Our framework consists of three main components:

- **Causal Representation Prototype Learning (CRP):** Uses adversarial deconfounding to suppress shortcut features and learn reliable causal prototypes for base classes.
- **Causal Reasoning Graph (CRG):** Models relationships between base and novel class prototypes through a learnable adjacency matrix, with **causal pruning** and **inference direction consistency** constraints.
- **Graph Convolutional Pseudo-label Generation (GCPL):** Propagates information over the refined graph to improve pseudo-labels for novel classes.

## Getting Started

**Installation.** Clone this repository and install the dependencies required by your PyTorch / PyTorch Lightning environment, including MinkowskiEngine for the sparse 3D backbone.

```bash
git clone https://github.com/Brucely674/Causal-3D-NCDSS.git
cd Causal-3D-NCDSS
```

**Training.** The repository provides `train.py` as the training entry point. For example, after preparing the dataset and its configuration file:

```bash
python train.py --dataset SemanticPOSS --split 0 --dataset_config path/to/semposs_dataset.yaml
```

**Evaluation.** Validation mIoU is logged during training, and the best checkpoint is saved based on `valid/mIoU`. See `utils/` for the evaluation callback.

Our experiments build on the 3D novel class discovery research literature and use the public SemanticKITTI and SemanticPOSS datasets. We thank the authors and maintainers of the relevant datasets and open-source frameworks.
