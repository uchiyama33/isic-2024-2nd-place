import hydra
import os
from lightning import LightningModule, LightningDataModule
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import tempfile
from tqdm.notebook import tqdm


def inference(rank, world_size, temp_dir, ckpt_paths, cfg, batch_size_pred):
    print(f"Running inference on rank {rank}.")

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("predict")
    dataset = datamodule.data_pred

    # データセットとデータローダーの設定
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size_pred, sampler=sampler, num_workers=2)

    # モデルの設定
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = model.to(rank).eval()
    if cfg.model.compile:
        model.net = torch.compile(model.net)

    for fold, ckpt_path in enumerate(ckpt_paths):
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        net = model.net

        # 推論の実行
        net.eval()
        all_predictions = []
        ids = []
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                for data in tqdm(dataloader, desc=f"rank_{rank}, fold_{fold}"):
                    if cfg.model.get("use_image") and cfg.model.get("use_metadata"):
                        logits = net(data["image"].to(rank), data["metadata"].to(rank))
                    else:  # elif cfg.model.use_image:
                        logits = net(data["image"].to(rank))
                    preds = torch.softmax(logits, dim=1)[:, 1]
                    all_predictions.append(preds.cpu())
                    ids.append(data["isic_id"])

        all_predictions = torch.cat(all_predictions)
        ids = np.concatenate(ids)

        # 推論結果をファイルに保存
        temp_file = os.path.join(temp_dir, f"predictions_rank_{rank}_fold_{fold}.pt")
        torch.save(all_predictions, temp_file)
        temp_file_id = os.path.join(temp_dir, f"ids_rank_{rank}_fold_{fold}.npy")
        np.save(temp_file_id, ids)

        print(
            f"Rank {rank}, fold {fold} finished inference with predictions saved to {temp_file}, {temp_file_id}"
        )
