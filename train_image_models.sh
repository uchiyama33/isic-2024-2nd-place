#!/bin/bash

cd src

experiments=(
    "0821-convnextv2_tiny-meta_target03-transV2-lr1e-3-target_decay001-bs128_2-ep100-neg5-tsgkf"
    "0821-eva02_small-sep_head-transV6-lr1e-3-target_decay001-warmup50-wd1e-2-drop01-bs32_8-ep80-neg3-tsgkf"
    "0821-beitv2_base-sep_head-transV8-lr1e-3-target_decay001-warmup50-wd1e-2-bs32_8-mixup-ep100-neg3-tsgkf"
    "0824-swinv2_small-transV2-lr1e-3-target_decay001-bs32_8-drop01-ep200-neg3-cluster7t5-tsgkf"
    "0824-eva02_small-sep_head-transV8-lr1e-3-target_decay0008-warmup50-wd1e-2-drop01-bs32_8-ep80-neg3-cluster7t5-tsgkf"
    "0827-deit3_small-transV2-lr1e-3-target_decay001-bs32_8-ep200-neg3-tsgkf"
    "0828-resnext50-transV2-lr1e-3-target_decay001-bs32_8-ep200-neg3-cluster7t5-tsgkf"
)

for experiment in "${experiments[@]}"; do
    python train_cv.py experiment=$experiment
    python save_train_predictions.py experiment=$experiment
    python train_all_data.py experiment=$experiment
done

experiments_tip_pretrain=(
    "0825-tip_pretrain-convnextv2_nano_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf"
    "0827-tip_pretrain-swin_tiny_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf"
)
for experiment in "${experiments_tip_pretrain[@]}"; do
    python train_cv_tip_pretrain.py experiment=$experiment
done

experiments_tip_finetune=(
    "0825-tip_finetune_onlyImage-convnextv2_nano_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs64_2-transV2-ep80"
    "0827-tip_finetune_onlyImage-swin_tiny_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs_64_2-transV8-ep80"
)

for experiment in "${experiments_tip_finetune[@]}"; do
    python train_cv.py experiment=$experiment
    python save_train_predictions.py experiment=$experiment
    python train_all_data.py data=isic2024_tip_finetune_train_all_data experiment=$experiment
done