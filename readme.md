# PPEL

Pathway-conditioned Prognostic Evidence Learning for multi-modal cancer survival analysis.

## Structure

```
├── main.py                    # Training & testing entry point
├── plot_km_curves.py          # Kaplan-Meier curve plotting
├── scripts/
│   ├── train.sh
│   └── test.sh
├── datasets/
│   └── dataset_survival.py    # SurvivalDatasetFactory & SurvivalDataset
├── models/
│   ├── model_PPEL.py          # PPEL model
│   └── omics_encoder.py       # Genomics encoder
├── utils/
│   ├── core_utils.py          # Training/validation loops
│   ├── general_utils.py       # Experiment setup, dataloaders
│   ├── loss_func.py           # Survival loss functions
│   ├── process_args.py        # Argument parsing
│   └── valid_utils.py         # Test evaluation
├── datasets_csv/              # Clinical data, metadata, RNA data
└── splits/                    # 5-fold cross-validation splits
```

## Supported Datasets

| Study | Cancer Type |
|--------|-------------|
| `tcga_blca` | Bladder |
| `tcga_brca` | Breast |
| `tcga_coadread` | Colorectal |
| `tcga_hnsc` | Head & Neck |
| `tcga_stad` | Stomach |

## Requirements

```bash
conda create -n ppel python=3.9
conda activate ppel
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Training

```bash
bash scripts/train.sh                        # defaults: tcga_stad, combine, swin
bash scripts/train.sh tcga_brca combine swin 0
```

Key arguments (see `process_args.py` for full list):

| Argument | Default | Description |
|-----------|---------|-------------|
| `--study` | `tcga_stad` | Dataset |
| `--type_of_path` | `combine` | Pathway set: xena / hallmarks / combine |
| `--mode` | `swin` | Feature extractor: swin / resnet50 / cluster |
| `--use_pel` / `--use_mel` / `--use_cgf` | `1` | PPEL components |
| `--sigma` / `--theta` | `0.1` | Loss weights |

## Testing

Evaluates with missing genomics modality:

```bash
bash scripts/test.sh ./results/PPEL/your_experiment
```

## Plotting KM Curves

```bash
python plot_km_curves.py --results_dir ./results/PPEL/your_experiment
```
