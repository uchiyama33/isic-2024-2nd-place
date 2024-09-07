ex="0904-9NNs-7types-feV7-s10-tuning_weights"
# cd /workspace/logs/train/runs
# cd /workspace/logs/train_all_data/runs
cd /workspace/logs/gbdt/runs
# cd /workspace/logs/gbdt_stacking/runs
zip -r ${ex}.zip ${ex} -x "${ex}/train_predictions/*" "${ex}/wandb/*"

