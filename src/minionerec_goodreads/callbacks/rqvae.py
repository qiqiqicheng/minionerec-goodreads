from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from minionerec_goodreads.utils.rqvae_stats import compute_sid_health_report


class RQVAEFullCatalogCheckpointCallback(L.Callback):
    def __init__(
        self,
        start_epoch: int | None,
        dirpath: str,
        filename: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        use_sk_infer: bool,
        save_weights_only: bool = True,
    ):
        super().__init__()
        self.start_epoch = start_epoch
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_sk_infer = use_sk_infer
        self.save_weights_only = save_weights_only

    def _enabled(self, trainer: L.Trainer) -> bool:
        return self.start_epoch is not None and not trainer.sanity_checking and trainer.current_epoch >= self.start_epoch

    def _infer_full_catalog_codes(self, pl_module: L.LightningModule, dataset: Any) -> torch.Tensor:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        codes = []
        was_training = pl_module.rqvae.training
        pl_module.rqvae.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(pl_module.device)  # [B, D]
                batch_codes = pl_module.rqvae.get_indices(batch, use_sk=self.use_sk_infer)  # [B, C]
                codes.append(batch_codes.cpu())
        if was_training:
            pl_module.rqvae.train()
        return torch.cat(codes, dim=0)  # [N, C]

    def _build_metrics(self, pl_module: L.LightningModule, codes: torch.Tensor) -> dict[str, float]:
        report = compute_sid_health_report(
            codes.numpy(),
            pl_module.rqvae.rq.codebook_size_list,
            code_offset=0,
            top_k=0,
            include_histograms=False,
        )
        metrics = {
            "val/full_catalog_collision_rate": float(report["paths"]["collision_rate"]),
            "val/full_catalog_unique_code_path_ratio": float(report["paths"]["unique_code_path_ratio"]),
            "val/full_catalog_num_unique_code_paths": float(report["paths"]["num_unique_code_paths"]),
            "val/full_catalog_num_collisions": float(report["paths"]["num_collisions"]),
        }
        for level in report["levels"]:
            level_idx = level["level"]
            metrics[f"val/full_catalog_used_ratio_l{level_idx}"] = float(level["used_ratio"])
            metrics[f"val/full_catalog_perplexity_l{level_idx}"] = float(level["perplexity"])
            metrics[f"val/full_catalog_used_codes_l{level_idx}"] = float(level["used_codes"])
            metrics[f"val/full_catalog_dead_codes_l{level_idx}"] = float(level["dead_codes"])
        return metrics

    def _checkpoint_path(self, trainer: L.Trainer) -> Path:
        filename = self.filename.format(epoch=trainer.current_epoch)
        if not filename.endswith(".ckpt"):
            filename = f"{filename}.ckpt"
        return self.dirpath / filename

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._enabled(trainer):
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("Full-catalog SID metrics require a datamodule")
        dataset = getattr(datamodule, "dataset", None)
        if dataset is None:
            raise RuntimeError("Call datamodule.setup() before full-catalog SID metrics")

        codes = self._infer_full_catalog_codes(pl_module, dataset)
        metrics = self._build_metrics(pl_module, codes)
        pl_module.log_dict(metrics, prog_bar=False, sync_dist=True)

        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(self._checkpoint_path(trainer), weights_only=self.save_weights_only)
