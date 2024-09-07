cd src

# python train_cv.py experiment=0825-tip_finetune_onlyImage-convnextv2_nano_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs64_2-transV2-ep80
# python train_cv.py experiment=0827-tip_finetune_onlyImage-swin_tiny_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs_64_2-transV8-ep80

# python save_train_predictions.py experiment=0827-tip_finetune_onlyImage-swin_tiny_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs_64_2-transV8-ep80
# python save_train_predictions.py experiment=0825-tip_finetune_onlyImage-convnextv2_nano_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs64_2-transV2-ep80

python train_all_data.py data=isic2024_tip_finetune_train_all_data experiment=0827-tip_finetune_onlyImage-swin_tiny_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs_64_2-transV8-ep80
python train_all_data.py data=isic2024_tip_finetune_train_all_data experiment=0825-tip_finetune_onlyImage-convnextv2_nano_scratch_l2d192h8-tabV3-transV8-lr5e-4-warmup50-bs_64_8-neg50-ep200-tsgkf-lr1e-3-warmup5-bs64_2-transV2-ep80
# python train_all_data.py experiment=0828-deit3_small-transV2-lr1e-3-target_decay001-bs32_8-ep200-neg3-cluster7t5-tsgkf
# python train_all_data.py experiment=0828-resnext50-transV2-lr1e-3-target_decay001-bs32_8-ep200-neg3-cluster7t5-tsgkf