# ISIC 2024: 2nd Place

## Environment
This solution uses the following Docker image:
 - gcr.io/kaggle-gpu-images/python:v150

All required Python packages are listed in the `pip_packages/requirements.txt` file.

## Data
- Place the competition dataset in the `data/` directory.
- Run the data preparation script:
  ```
  cd src
  python prepare_data.py
  ```

## Training Image Models
  ```
  bash train_image_models.sh
  ```

## Training GBDT Models
  ```
  cd src
  python gbdt.py
  ```

## Trained Weights
[https://www.kaggle.com/datasets/tomoon33/isic2024-training-logs/](https://www.kaggle.com/datasets/tomoon33/isic2024-training-logs/)

## Inference

The submission notebook is `isic2024-submit.ipynb`.

## Acknowledgements
We extend our sincere gratitude to the creators of the following repositories, whose outstanding work significantly contributed to our project:

 - [ashleve/lightning-hydra-template: PyTorch Lightning + Hydra. A very user-friendly template for ML experimentation. ⚡🔥⚡](https://github.com/ashleve/lightning-hydra-template)
  - [siyi-wind/TIP: [ECCV 2024] TIP: Tabular-Image Pre-training for Multimodal Classification with Incomplete Data (an official implementation)](https://github.com/siyi-wind/tip)
