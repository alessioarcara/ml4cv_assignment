# Machine Learning for Computer Vision Assignment
A.Y. 2024–2025.\
alessio.arcara@studio.unibo.it

---

## Overview 

Standard segmentation models are trained on a closed set of classes, forcing them to categorize every pixel into a known label. Consequently, when they encounter an unknown object, they erroneously classify it as a known class. This project tackles **Open-Set Semantic Segmentation** for autonomous driving to correct this behavior. The goal is to detect unexpected hazards (such as animals or lost cargo) by accurately identifying objects that fall outside the training distribution, while maintaining high-quality segmentation masks for known classes.

<div align="center">
  <p>An example of open-set segmentation</p>
  <img src="assets/both.png" alt="Open-set segmentation" width="700"/>
</div>

## Installation

### 1. **Clone the repository:**

```bash
git clone https://github.com/alessioarcara/ml4cv_assignment
cd ml4cv_assignment
```

### 2. **Set up the environment:**

You can set up the environment using `uv` or standard `pip`.

#### Option A: Using uv (recommended)

```bash
uv venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
uv sync
```

#### Option B: Using pip

This project adheres to PEP 621 standards using `pyproject.toml`.

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e .
```

### 3. **Dataset setup:**

The dataset must be downloaded manually from the links provided above and extracted into the `datasets/` folder. Create the folder if it does not exist. 

#### A. Street Hazards (Primary)

1. Download:
    * [Training and validation sets download link](https://people.eecs.berkeley.edu/~hendrycks/streethazards_train.tar)
    * [Test set download link](https://people.eecs.berkeley.edu/~hendrycks/streethazards_test.tar)

2. Extract the contents into the `datasets/train/` and `datasets/test/` folders respectively.

#### B. Road Anomaly Benchmark (Optional/Benchmarking)

Required only if you intend to run the `benchmark_smiyc` benchmark script.

1. Download the **RoadAnomaly21** from the [SegmentMentMeIfYouCan website](https://segmentmeifyoucan.com/datasets).
2. Extract the contents into the `datasets/dataset_AnomalyTrack/` folder.

#### C. Cityscapes (Optional/Pre-training)

Used for closed-set and open-set trainings (with OutlierExposure) before RoadAnomaly evaluation.

1. Register from the [official website](https://www.cityscapes-dataset.com/downloads/) to dowload the Cityscapes dataset.
2. Download the following packages:
   * `leftImg8bit_trainvaltest.zip` (Images)
   * `gtFine_trainvaltest.zip` (Annotations)
3. Extract the contents into the `datasets/cityscapes/` folder.

The resulting structure should look like this:

```plain
datasets/
├── test/                   # Street Hazards (Test)
├── train/                  # Street Hazards (Train)
├── RoadAnomaly/            # Road Anomaly Benchmark
└── cityscapes/             # Cityscapes
    ├── gtFine/
    └── leftImg8bit/
```

### 4. **Download weights (optional):**

All pre-trained models are available for download in the table below. Once downloaded, please place the files into the `checkpoints/` directory.

| Model Name | Download Link |
| :--- | :---: |
| **ProtoSegNet (Baseline)** | [weights](https://drive.google.com/file/d/15fM3p8VJRrLllwiO7Fu7f-O7N1LuGaqR/view?usp=share_link) |
| **ProtoSegNet (Optimized)** | [weights](https://drive.google.com/file/d/1WaaYNKvnsZiBZ98xj5zM-Nawzo78xDGH/view?usp=share_link) |
| **ProtoSegNet (Cityscapes)** | [weights](https://drive.google.com/file/d/1PD0exlLyXCLoDQhAznQthKZCLEaUSHlg/view?usp=share_link) |
| **Mask2Former + RbA** | [weights](https://drive.google.com/file/d/16Js6iKtZAV30Mj56064mOYSVloMTtAOp/view?usp=share_link) |

Alternatively, you can use the provided script to download all checkpoints automatically:

```bash
uv run download_checkpoints.py
```

## Usage

All experiments were conducted locally on a dedicated machine with an **NVIDIA RTX 3090 GPU**.

This project extensively uses configuration files to manage experiments. To see how to use configurations, please refer to the `configs/` folder. You can compose multiple config files to override specific parameters.

### Training a model

To train a model, use the provided training script. You can pass one or multiple configuration files:

```bash
uv run python scripts/train.py --configs configs/base.yaml configs/experiment_1.yaml
```

**Arguments:**
* `--configs`: One or more paths to YAML config files (space-separated).

For extensive ablation studies, the `runner.sh` script is available as reference. It automates the execution of `train.py` across multiple experiments:

```bash
bash runner.sh
```

### Evaluating a trained model

To evaluate a model, use the `eval.py` script:

```bash
uv run python scripts/eval.py \
    --configs configs/base.yaml configs/experiment_1.yaml \
    --split test \
    --checkpoint_path output/checkpoints/best_model.pth
```

**Arguments:**
* `--configs`: One or more paths to YAML config files (space-separated).
* `--split`: The dataset split to evaluate on (test or val). Defaults to test.
* `--checkpoint_path`: (Optional) Path to a specific .pth model file. If not provided, the script will look for the default checkpoint defined in your config.

### Benchmarking on RoadAnomaly (SegmentMeIfYouCan)

To benchmark the model on the RoadAnomaly dataset, use the `benchmark_smiyc.py` script:

```bash
uv run python scripts/benchmark_smiyc.py \
    --dataset AnomalyTrack-all
```

**Note**: This script requires the `RoadAnomaly` dataset to be correctly placed in `datasets/dataset_AnomalyTrack/` as described in the Dataset setup section.

## Project Structure

```plain
ml4cv_assignment/
├── assets/             # Images for README/Notebooks
├── checkpoints/        # Model checkpoints
├── configs/            # YAML experiment configurations 
│   ├── base.yaml       # Base configuration
│   └── ...
├── datasets/           # Dataset directory
│   ├── test/           # Extracted from streethazards_test.tar
│   ├── train/          # Extracted from streethazards_train.tar
│   ├── dataset_AnomalyTrack/  # Road Anomaly Benchmark
│   └── cityscapes/     # Cityscapes dataset
├── external/           # Submodules
│   └── road-anomaly-benchmark/
├── ml4cv_assignment/   # Main package
│   ├── config/         # Pydantic schemas and configuration logic
│   ├── data/           # Dataset, augmentations and collation
│   ├── models/         # Network architectures
│   ├── training/       # Training loop, losses, metrics and callbacks
│   └── utils/          # Utilities (Visualization, Logging, IO)
├── notebooks/          
│   ├── main.ipynb      # 📄 MAIN REPORT
├── scripts/            # Entry points
│   ├── eval.py         # Evaluation script
│   └── train.py        # Training script
├── runner.sh           # Bash script for batch experiments
├── pyproject.toml      # Project dependencies and metadata
└── README.md           # Project setup instructions
```