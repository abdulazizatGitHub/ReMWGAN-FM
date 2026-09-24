# ReMWGAN-FM: Reweighted Multi-Generator Wasserstein GAN with Feature Matching

This repository implements a class-conditional generative adversarial network pipeline for tabular network intrusion data, with a focus on the UNSW-NB15 dataset. The project explores how synthetic data generation can rebalance minority classes and improve downstream intrusion-detection classification performance.

At a high level, the project trains multiple GAN variants, generates class-balanced synthetic samples, and evaluates the impact on classification accuracy and F1-score. The code is structured as a research pipeline rather than a packaged application, and it contains both training-time generative models and separate classifier evaluation scripts.

---

## 1. Project Objective

The main goal is to improve the quality of network intrusion detection datasets by generating realistic samples for underrepresented attack classes.

The repository focuses on these ideas:

- class imbalance reduction for multi-class intrusion detection
- synthetic sample generation for minority classes using GANs
- comparison of baseline GAN, Wasserstein GAN, and custom ReMWGAN-FM variants
- classification evaluation after augmentation
- visualization of generated distributions and training behavior

The dataset used in the project is UNSW-NB15, which is a benchmark intrusion detection corpus containing normal traffic and multiple attack categories.

---

## 2. Core Idea of the Codebase

The project is built around a repeated pattern:

1. Load and preprocess the UNSW-NB15 dataset.
2. Split samples into train/test sets.
3. Train one or more label-aware generators.
4. Collect generated samples for each class.
5. Balance the dataset by filling smaller classes up to the largest class count.
6. Save the synthetic or augmented dataset.
7. Train or evaluate a classifier using the augmented samples.

This pattern is implemented in both the training modules and the classification scripts.

The repo mainly centers on a custom model called ReMWGAN-FM (Reweighted Multi-Generator Wasserstein GAN with Feature Matching), with an enhanced DPL variant used to improve class-aware generation and balancing.

---

## 3. Main Models in This Repo

### 3.1 Original TMG-GAN baseline

File: [training/tmg_gan_reference.py](training/tmg_gan_reference.py)

This is the original/reference TMG-GAN implementation. It is a baseline class-conditioned GAN that learns per-class generators and uses a shared discriminator/classifier head.

Key characteristics:

- one generator per class
- one shared classifier/discriminator model
- class-aware adversarial training
- cosine-similarity-based comparison between real and generated feature embeddings
- training saves models and logs metrics

This file represents the original TMG-GAN approach and is not the custom ReMWGAN-FM model name used in this project.

### 3.2 Our method: ReMWGAN-FM

File: [training/remwgan_fm.py](training/remwgan_fm.py)

This is the implementation that belongs to the project’s custom method: ReMWGAN-FM. It is the method we built for this project and is the one that includes the reweighting logic, feature-matching loss, and imbalance-aware training behavior.

Important ideas in this version:

- WGAN critic objective
- gradient penalty
- class-weight adjustments based on class imbalance ratios
- feature matching between real and fake samples
- scheduler-based optimization
- output generation for visualization and model saving

This is the actual custom method used in this repo and should be treated as the project’s own implementation.

### 3.3 ReMWGAN-FM Dynamic

File: [training/TMG_GAN_Dynamic.py](training/TMG_GAN_Dynamic.py)

This file is a compatibility extension wrapper over the custom ReMWGAN-FM implementation and is kept for dynamic-training compatibility.

This variant is designed to work with dynamically provided original GAN samples and is used during the balanced-data generation workflow. It is a dynamic extension of the custom ReMWGAN-FM pipeline and emphasizes class-specific learning under imbalance.

Important ideas in this version:

- WGAN critic objective
- gradient penalty
- class-weight adjustments based on class imbalance ratios
- feature matching between real and fake samples
- scheduler-based optimization
- output generation for visualization and model saving

This is the closest thing to the final “proposed method” in the project.

### 3.4 Baseline GAN variants

Files:

- [training/GAN.py](training/GAN.py)
- [training/WGAN.py](training/WGAN.py)
- [training/SNGAN.py](training/SNGAN.py)

These are baseline implementations used to compare simpler GAN designs against the original TMG-GAN reference and the custom ReMWGAN-FM method.

---

## 4. Dataset and Preprocessing

The project is based on UNSW-NB15, and the data files are expected under:

- [data/datasets](data/datasets)

The most relevant dataset files are:

- UNSW_NB15_training-set.csv
- UNSW_NB15_testing-set.csv
- filtered variants for the selected attack categories

### 4.1 Label Set Used

The preprocessing logic in [training/datasets/__init__.py](training/datasets/__init__.py) selects five categories:

- Normal
- DoS
- Reconnaissance
- Shellcode
- Worms

These are mapped to integer labels 0 through 4.

### 4.2 Feature Engineering

The processing pipeline includes:

- categorical columns: proto, service, state
- numerical columns: traffic statistics, connection lengths, packet counts, etc.
- MinMaxScaler for numerical features
- OneHotEncoder for categorical features
- SelectKBest feature selection with k = 55

This is a strong indicator that the project is designed to work with mixed-type tabular network traffic features.

The preprocessing is built to transform UNSW-NB15 data into a compact, model-friendly feature vector for GAN training.

---

## 5. Repository Structure

The repository is organized as follows:

- [training](training) — core PyTorch training code and model implementations
- [scripts](scripts) — training entry points and dataset preparation scripts
- [classification](classification) — classifier experiments and result plots
- [testing](testing) — model sanity tests and visualization scripts
- [data](data) — datasets, generated outputs, and saved models
- [requirements.txt](requirements.txt) — Python dependencies

### Main directories

#### training/

This is the heart of the project.

- [training/GAN.py](training/GAN.py): vanilla conditional GAN baseline
- [training/WGAN.py](training/WGAN.py): Wasserstein GAN baseline
- [training/SNGAN.py](training/SNGAN.py): spectral-normalized GAN baseline
- [training/TMG_GAN.py](training/TMG_GAN.py): main class-conditional TMG-GAN
- [training/TMG_GAN_DPL.py](training/TMG_GAN_DPL.py): advanced DPL variant
- [training/models](training/models): generator and discriminator architectures
- [training/config](training/config): training hyperparameters and output paths
- [training/datasets](training/datasets): dataset wrappers and preprocessing metadata

#### scripts/

- [scripts/train_tmg_gan.py](scripts/train_tmg_gan.py): trains the base TMG-GAN and dynamic TMG-GAN, then saves balanced datasets
- [scripts/train_tmg_gan_dpl.py](scripts/train_tmg_gan_dpl.py): trains the DPL TMG-GAN variant
- [scripts/examine_dataset.py](scripts/examine_dataset.py): inspects and filters dataset categories
- [scripts/visualize_gan_output.py](scripts/visualize_gan_output.py): plotting utilities for GAN output visualization
- [scripts/run.sbatch](scripts/run.sbatch): Slurm GPU job script for cluster execution

#### classification/

The classification scripts train downstream models on the generated balanced datasets.

- [classification/classifier.py](classification/classifier.py): multi-class classifier pipeline
- [classification/classifier_binary.py](classification/classifier_binary.py): binary classifier variant

These scripts load pickle files created by GAN training, then evaluate accuracy, precision, recall, F1, and sometimes confusion matrices and ROC plots.

#### data/

This folder stores the actual artifacts:

- datasets/
- GAN_output/
- logs/
- Results/
- trained_models/

The saved model snapshots are stored under:

- [data/trained_models](data/trained_models)

---

## 6. How Training Works

### 6.1 Dataset object and class grouping

The training code uses `Dataset` wrappers from:

- [training/datasets/_dataset.py](training/datasets/_dataset.py)
- [training/datasets/tr_dataset.py](training/datasets/tr_dataset.py)
- [training/datasets/te_dataset.py](training/datasets/te_dataset.py)

These objects group samples by class label and provide access to the train/test tensors used by the GANs.

### 6.2 Generator architecture

File: [training/models/generator_model.py](training/models/generator_model.py)

The generator is a multi-layer MLP:

- input: latent vector of size 128
- hidden layers: 512 -> 128 -> 32
- output: feature vector matching the dataset dimensionality
- final activation: Sigmoid

This is appropriate for tabular feature generation and normalized network-flow data.

### 6.3 Discriminator/classifier architecture

File: [training/models/cd_model.py](training/models/cd_model.py)

The discriminator/classifier model has two output heads:

- critic score: realness signal for GAN loss
- class logits: classification output for the target label

This dual-head architecture is a core design choice in the repo and is what allows the model to combine adversarial realism learning with supervision.

---

## 7. Training Flow in Practice

The following flow is the most representative of the repository's intended methodology:

1. Load training data
2. Prepare class-wise samples
3. Train a label-aware GAN
4. Generate additional samples for underrepresented classes
5. Balance each class to the maximum class count
6. Save the augmented dataset as a pickle file
7. Train a downstream classifier on the synthetic+real data
8. Measure accuracy, precision, recall, and F1

This is visible in scripts such as:

- [scripts/train_tmg_gan.py](scripts/train_tmg_gan.py)
- [scripts/train_tmg_gan_dpl.py](scripts/train_tmg_gan_dpl.py)

### Example of the balancing logic

The script computes the largest class count and fills all other classes up to that count by generating new samples. The generated samples are then appended to the original training set, and the resulting augmented dataset is saved.

This is the main mechanism used to counter class imbalance.

---

## 8. Key Output Artifacts

The project saves multiple kinds of artifacts during training:

### Training outputs

- loss plots
- generated sample visualizations
- logs with epoch-by-epoch metrics
- saved PyTorch model weights

Typical paths:

- [data/GAN_output/ReMWGAN_FM](data/GAN_output/ReMWGAN_FM)
- [data/GAN_output/ReMWGAN_FM_Dynamic](data/GAN_output/ReMWGAN_FM_Dynamic)
- [data/GAN_output/DPL_ReMWGAN_FM](data/GAN_output/DPL_ReMWGAN_FM)

### Model checkpoints

- [data/trained_models](data/trained_models)

These contain model states for:

- discriminator/classifier
- per-class generator models

### Classification outputs

The experiment scripts save confusion matrices, ROC curves, and metric summaries into:

- [classification/ReMWGAN_FM_results](classification/ReMWGAN_FM_results)

---

## 9. How to Run the Project

### 9.1 Install dependencies

Use the repo's requirements file:

```bash
pip install -r requirements.txt
```

### 9.2 Prepare the dataset

The project expects the dataset to already be present in the data folder.

You can inspect and filter it with:

```bash
python scripts/examine_dataset.py
```

### 9.3 Train the original TMG-GAN baseline pipeline

```bash
python scripts/train_tmg_gan_baseline.py
```

This script trains the original TMG-GAN baseline and the dynamic variant, then saves balanced datasets as pickle files.

### 9.4 Train our custom ReMWGAN-FM method

```bash
python scripts/train_remwgan_fm.py
```

This script trains the custom ReMWGAN-FM method and saves generated samples and models.

### 9.5 Run classification evaluation

```bash
python classification/classifier.py
```

This reads the balanced GAN-generated dataset and evaluates performance on the test set.

---

## 10. Important Configuration

The training hyperparameters are stored in:

- [training/config/GAN_config.py](training/config/GAN_config.py)

Key defaults:

- epochs: 2000
- batch_size: 128
- z_size: 128
- g_lr: 2e-4
- cd_lr: 2e-4
- g_loopNo: 1
- cd_loopNo: 5

The output directories are defined in:

- [training/config/path_config.py](training/config/path_config.py)

These paths ensure model checkpoints and plots go to consistent output folders under the data directory.

---

## 11. Diagnostic and Visualization Scripts

The repo includes several analysis utilities:

- [scripts/visualize_gan_output.py](scripts/visualize_gan_output.py): GAN output plotting utilities
- [testing/test_tmg_gan.py](testing/test_tmg_gan.py): visual and validation script for ReMWGAN-FM output
- [testing/test_gan.py](testing/test_gan.py): general GAN sanity-check test

These scripts help check whether the generators are producing realistic samples and whether the learned distributions are aligned with the real data.

---

## 12. Research Interpretation

This repo is best understood as a research implementation for data augmentation in network intrusion detection.

The project tries to answer this question:

> Can a class-conditioned GAN generate realistic minority-class attack samples to improve imbalance handling and overall classification performance?

The architecture is designed to answer that by combining:

- adversarial training
- class supervision
- feature matching and cosine geometry
- target-class balancing

In other words, the system is not merely producing random fake traffic; it is specifically learning per-class distributions and generating samples to address underrepresentation.

---

## 13. Strengths of the Project

- research-oriented GAN workflow tailored to tabular intrusion detection data
- multi-class class-aware generation instead of naive oversampling
- explicit balancing of minority classes using generated data
- direct downstream evaluation with classifier metrics
- code organized around reproducible training and output artifacts

---

## 14. Limitations and Caveats

This repo is a research prototype, and there are a few important caveats:

1. Some scripts appear to be experimental or partially outdated.
2. The project mixes model training, data synthesis, and classifier evaluation in a compact but somewhat loosely connected pipeline.
3. The project assumes a specific UNSW-NB15 preprocessing scheme; mismatches in feature order or encoding can break model compatibility.
4. The README and code structure suggest active experimentation, so not every script is production-ready.
5. There are traces of older or alternate dataset preprocessing methods; consistency matters when reusing saved models.

This means the repository is best treated as an academic or experimental codebase rather than a polished end-user application.

---

## 15. Clear Understanding of the Repo

The project is fundamentally a GAN-based augmentation pipeline for intrusion detection.

The main conceptual flow is:

- intrusion data is imbalanced
- the original TMG-GAN baseline is used as a reference model
- the custom ReMWGAN-FM method learns class-specific feature distributions with reweighting and feature matching
- synthetic samples are generated for minority attack classes
- the augmented dataset is used to train a classifier
- model quality is measured using standard classification metrics

The most important files to understand first are:

1. [training/datasets/__init__.py](training/datasets/__init__.py) — data setup and label mapping
2. [training/models/generator_model.py](training/models/generator_model.py) — generator architecture
3. [training/models/cd_model.py](training/models/cd_model.py) — critic/classifier shared model
4. [training/tmg_gan_reference.py](training/tmg_gan_reference.py) — original TMG-GAN reference implementation
5. [training/remwgan_fm.py](training/remwgan_fm.py) — custom ReMWGAN-FM implementation
6. [scripts/train_tmg_gan_baseline.py](scripts/train_tmg_gan_baseline.py) — baseline and dynamic training flow
7. [scripts/train_remwgan_fm.py](scripts/train_remwgan_fm.py) — custom ReMWGAN-FM training flow
8. [classification/classifier.py](classification/classifier.py) — downstream evaluation pipeline

These files together describe the project’s actual scientific and engineering workflow.

---

## 16. Recommended Reading Order

If you want to understand the repo quickly, read in this order:

1. [training/datasets/__init__.py](training/datasets/__init__.py)
2. [training/models/generator_model.py](training/models/generator_model.py)
3. [training/models/cd_model.py](training/models/cd_model.py)
4. [training/TMG_GAN.py](training/TMG_GAN.py) — original baseline
5. [training/TMG_GAN_DPL.py](training/TMG_GAN_DPL.py) — custom ReMWGAN-FM method
6. [scripts/train_tmg_gan_dpl.py](scripts/train_tmg_gan_dpl.py) — custom training pipeline
7. [classification/classifier.py](classification/classifier.py) — downstream evaluation

That path covers the full lifecycle from dataset preparation to augmentation to performance evaluation.

---

## 17. Summary

This repository is a research implementation of class-aware GAN-based data augmentation for imbalanced intrusion-detection datasets. It combines custom GAN architecture, dataset balancing, and downstream classification evaluation to demonstrate how synthetic network traffic can support improved detection of minority classes.

The repo is strongest when viewed as a training and experimentation pipeline for intrusion detection research, not as a general-purpose application framework.
