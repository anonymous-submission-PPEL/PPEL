# PPEL

Pathway-conditioned Prognostic Evidence Learning for multi-modal cancer survival analysis.

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

## Testing

Evaluates with missing genomics modality:

```bash
bash scripts/test.sh ./results/PPEL/your_experiment
```

