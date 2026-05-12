from typing import Any, Callable

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from minionerec_goodreads.basic.activation import activation_layer
from minionerec_goodreads.utils.rqvae_stats import compare_code_assignments, compute_sid_health_report


class MLP(nn.Module):
    def __init__(self, layers: list[int], dropout: float, activation: str = "relu", bn: bool = False):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.bn = bn

        mlps = []
        for i, (input_dim, output_dim) in enumerate(zip(layers[:-1], layers[1:])):
            mlps.append(nn.Dropout(self.dropout))
            mlps.append(nn.Linear(input_dim, output_dim))
            if self.bn and i != len(layers) - 2:  # exclude the last layer
                mlps.append(nn.BatchNorm1d(output_dim))

            activation_fn = activation_layer(self.activation)
            if activation_fn is not None and i != len(layers) - 2:  # exclude the last layer
                mlps.append(activation_fn)

        self.mlp_layers = nn.Sequential(*mlps)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, D_in]
        Returns:
            torch.Tensor: [B, D_out]
        """
        return self.mlp_layers(x)


class CodeBook(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        size: int,
        loss_beta: float,
        kmeans_init: bool,
        kmeans_iters: int,
        sk_temp: float,
        sk_iters: int = 100,
        kmeans_seed: int = 728,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.size = size
        self.loss_beta = loss_beta
        self.embeddings = nn.Embedding(self.size, self.emb_dim)
        self.register_buffer("initted", torch.tensor(not kmeans_init))
        self.sk_temp = sk_temp
        self.sk_iters = sk_iters
        self.kmeans_seed = kmeans_seed
        self.last_kmeans_stats: dict[str, Any] = {}

        if not kmeans_init:
            self.embeddings.weight.data.uniform_(-1.0 / self.size, 1.0 / self.size)
        else:
            if kmeans_iters <= 0:
                raise ValueError(f"kmeans_iters must be positive, got {kmeans_iters}")
            self.kmeans_iters = kmeans_iters

    def _kmeans_init(self, x: torch.Tensor) -> dict[str, Any]:
        """
        Args:
            x (torch.Tensor): [B, D_emb]
        """
        sample_count, _ = x.shape
        if sample_count < self.size:
            raise ValueError(f"K-means init needs at least {self.size} samples, got {sample_count}")
        device = x.device
        x_np = x.detach().cpu().numpy()
        unique_count = int(np.unique(x_np, axis=0).shape[0])
        cluster = KMeans(
            n_clusters=self.size,
            max_iter=self.kmeans_iters,
            n_init="auto",
            random_state=self.kmeans_seed,
        ).fit(x_np)
        centers = cluster.cluster_centers_  # [N, D_emb]
        self.embeddings.weight.data.copy_(torch.from_numpy(centers).to(device))
        self.initted.fill_(True)
        self.last_kmeans_stats = {
            "sample_count": int(sample_count),
            "duplicate_count": int(sample_count - unique_count),
            "inertia": float(cluster.inertia_),
        }
        return self.last_kmeans_stats

    @property
    def codebook(self) -> torch.Tensor:
        return self.embeddings.weight.data  # [N, D_emb]

    def get_codeword(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embeddings(indices)

    def sk_distance(self, d: torch.Tensor) -> torch.Tensor:
        max_d = d.max()
        min_d = d.min()
        middle = (max_d + min_d) / 2
        amplitude = max_d - middle + 1e-5
        centered_d = (d - middle) / amplitude
        return centered_d

    def sk_algorithm(self, d: torch.Tensor, temp: float, max_iters: int) -> torch.Tensor:
        B, N = d.shape
        Q = torch.exp(-d / temp)  # [B, N]
        Q = Q / Q.sum()  # [B, N]
        for _ in range(max_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= N  # column sum up to be 1/N

            Q /= Q.sum(dim=1, keepdim=True)
            Q /= B  # row sum up to be 1/B

        return Q

    def forward(self, x: torch.Tensor, use_sk: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [B, D_emb]
            use_sk (bool, optional): _description_. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - quantized (torch.Tensor): [B, D_emb]
                - quantization_loss (torch.Tensor): [,]
                - indices (torch.Tensor): [B, 1]
        """
        if not self.initted and self.training:
            self._kmeans_init(x)
        distances = (
            torch.sum(x**2, dim=1, keepdim=True)  # [B, 1]
            + torch.sum(self.embeddings.weight**2, dim=1, keepdim=True).t()  # [1, N]
            - 2 * torch.matmul(x, self.embeddings.weight.t())  # [B, N]
        )  # [B, N]

        if not use_sk:
            indices = torch.argmin(distances, dim=1)  # [B,]
        else:
            sinkhorn_distance = self.sk_distance(distances).double()  # [B, N]
            Q = self.sk_algorithm(sinkhorn_distance, temp=self.sk_temp, max_iters=self.sk_iters)  # [B, N]
            if torch.isnan(Q).any():
                raise ValueError(f"NaN values found in soft assignment matrix Q.:\n{Q}")
            indices = torch.argmax(Q, dim=1)  # [B,]

        x_q = self.embeddings(indices)  # [B, D_emb]
        commitment_loss = F.mse_loss(x_q.detach(), x)  # [,]
        codebook_loss = F.mse_loss(x_q, x.detach())  # [,]
        loss = commitment_loss + self.loss_beta * codebook_loss  # [,]

        # x_q = x + (x_q - x).detach()  # do this in rvq
        return x_q, loss, indices.unsqueeze(dim=1)  # [B, D_emb], [,], [B, 1]


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        loss_beta: float,
        codebook_size_list: list[int],  # e.g. [32, 64, 128]
        kmeans_init: bool,
        kmeans_iters: int,
        sk_temp: float,
        sk_iters: int = 100,
        kmeans_seed: int = 728,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.codebook_size_list = codebook_size_list
        self.loss_beta = loss_beta
        self.codebooks = nn.ModuleList([
            CodeBook(
                emb_dim=emb_dim,
                size=size,
                loss_beta=loss_beta,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                sk_temp=sk_temp,
                sk_iters=sk_iters,
                kmeans_seed=kmeans_seed + level_idx,
            )
            for level_idx, size in enumerate(self.codebook_size_list)
        ])

    def get_codebooks(self) -> list[torch.Tensor]:
        codebooks = []
        for cb in self.codebooks:
            codebooks.append(cb.codebook)  # [N_i, D_emb]
        return codebooks

    def needs_kmeans_init(self) -> bool:
        return any(not bool(cb.initted.item()) for cb in self.codebooks)

    @torch.no_grad()
    def kmeans_init(self, x: torch.Tensor) -> list[dict[str, Any]]:
        """
        Args:
            x (torch.Tensor): [B, D_emb]
        """
        reports = []
        res = x
        for level_idx, cb in enumerate(self.codebooks):
            report = {"level": level_idx, **cb._kmeans_init(res)}
            _, _, indices = cb(res, use_sk=False)  # [B, D_emb], [,], [B, 1]
            res = res - cb.get_codeword(indices.squeeze(dim=1))  # [B, D_emb]
            reports.append(report)
        return reports

    def forward(self, x: torch.Tensor, use_sk: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [B, D_emb]
            use_sk (bool): Whether to use the Sinkhorn-Knopp algorithm for soft assignment.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - quantized (torch.Tensor): [B, D_emb]
                - mean_losses (torch.Tensor): [,]
                - indices (torch.Tensor): [B, C]
        """
        all_losses = []
        all_indices = []
        x_q = torch.zeros_like(x)
        res = x
        for cb in self.codebooks:
            res_q, loss, indices = cb(res, use_sk)
            res = res - res_q  # [B, D_emb]
            x_q = x_q + res_q  # [B, D_emb]
            all_losses.append(loss)  # [,]
            all_indices.append(indices)  # [B, 1]

        mean_losses = sum(all_losses) / len(all_losses)  # [,]
        all_indices = torch.concat(all_indices, dim=1)  # [B, C]

        x_q_out = x + (x_q - x).detach()  # [B, D_emb] STE

        return x_q_out, mean_losses, all_indices


class RQVAE(nn.Module):
    def __init__(self, in_dim: int, layers: list[int], emb_dim: int, rq: ResidualVectorQuantizer):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.layers = layers

        self.mlp_layers = [self.in_dim] + self.layers + [self.emb_dim]

        self.encoder = MLP(self.mlp_layers, dropout=0.1, activation="relu", bn=True)
        self.rq = rq
        self.decoder = MLP(self.mlp_layers[::-1], dropout=0.1, activation="relu", bn=True)

    def forward(self, x: torch.Tensor, use_sk: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [B, D_in]
            use_sk (bool): Whether to use the Sinkhorn-Knopp algorithm for soft assignment.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_out (torch.Tensor): [B, D_in]
                - rq_loss (torch.Tensor): [,]
                - indices (torch.Tensor): [B, C]
        """
        x_encoded = self.encoder(x)  # [B, D_emb]
        x_q, rq_loss, indices = self.rq(x_encoded, use_sk)  # [B, D_emb], [,], [B, C]
        x_out = self.decoder(x_q)

        return x_out, rq_loss, indices

    @torch.no_grad()
    def init_codebooks(self, x: torch.Tensor) -> list[dict[str, Any]]:
        x_encoded = self.encoder(x)  # [B, D_emb]
        return self.rq.kmeans_init(x_encoded)

    @torch.no_grad()
    def get_indices(self, x: torch.Tensor, use_sk: bool = True) -> torch.Tensor:
        x_encoded = self.encoder(x)  # [B, D_emb]
        _, _, indices = self.rq(x_encoded, use_sk)  # [B, D_emb], [,], [B, C]

        return indices  # [B, C]


class RQVAEModule(L.LightningModule):
    def __init__(
        self,
        rqvae: RQVAE,
        optimizer: Callable,
        scheduler: Callable | None,
        use_sk: bool = True,
        kmeans_sample_size: int = 4096,
        kmeans_sample_seed: int = 728,
    ):
        super().__init__()
        self.rqvae = rqvae
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.use_sk = use_sk
        self.kmeans_sample_size = kmeans_sample_size
        self.kmeans_sample_seed = kmeans_sample_seed
        self._epoch_indices: dict[str, list[torch.Tensor]] = {}
        self._epoch_argmin_indices: dict[str, list[torch.Tensor]] = {}
        self._pending_kmeans_metrics: dict[str, float] = {}
        self.save_hyperparameters(logger=False, ignore=["rqvae"])

    def _validate_embedding_dim(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        dataset = getattr(datamodule, "dataset", None)
        if dataset is None:
            datamodule.setup("fit")
            dataset = getattr(datamodule, "dataset", None)
        if dataset is None:
            return
        actual_dim = int(dataset.embeddings.shape[1])
        if actual_dim != self.rqvae.in_dim:
            raise ValueError(f"RQVAE input dim mismatch: embedding dim {actual_dim} != model.rqvae.in_dim {self.rqvae.in_dim}")

    def _collect_kmeans_pool(self) -> torch.Tensor:
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            raise RuntimeError("RQVAE K-means initialization requires a Lightning datamodule")
        train_data = getattr(datamodule, "data_train", None)
        if train_data is None:
            datamodule.setup("fit")
            train_data = getattr(datamodule, "data_train", None)
        if train_data is None:
            raise RuntimeError("Call datamodule.setup('fit') before RQ-VAE K-means initialization")
        sample_count = min(self.kmeans_sample_size, len(train_data))
        generator = torch.Generator().manual_seed(self.kmeans_sample_seed)
        sample_indices = torch.randperm(len(train_data), generator=generator)[:sample_count].tolist()
        samples = torch.stack([train_data[index] for index in sample_indices], dim=0)  # [N, D_in]
        return samples.to(self.device)

    def _log_kmeans_reports(self, reports: list[dict[str, Any]]) -> None:
        self._pending_kmeans_metrics = {}
        for report in reports:
            level = report["level"]
            self._pending_kmeans_metrics[f"train/kmeans_sample_count_l{level}"] = float(report["sample_count"])
            self._pending_kmeans_metrics[f"train/kmeans_duplicate_count_l{level}"] = float(report["duplicate_count"])
            self._pending_kmeans_metrics[f"train/kmeans_inertia_l{level}"] = float(report["inertia"])

    def on_fit_start(self) -> None:
        self._validate_embedding_dim()
        if self.kmeans_sample_size <= 0 or not self.rqvae.rq.needs_kmeans_init():
            return
        sample_pool = self._collect_kmeans_pool()
        was_training = self.rqvae.training
        self.rqvae.eval()
        reports = self.rqvae.init_codebooks(sample_pool)
        if was_training:
            self.rqvae.train()
        self._log_kmeans_reports(reports)

    def _reset_code_metrics(self, stage: str) -> None:
        self._epoch_indices[stage] = []
        self._epoch_argmin_indices[stage] = []

    def _track_code_metrics(self, stage: str, batch: torch.Tensor, indices: torch.Tensor) -> None:
        self._epoch_indices.setdefault(stage, []).append(indices.detach().cpu())
        if stage in {"val", "test"} and self.use_sk:
            argmin_indices = self.rqvae.get_indices(batch, use_sk=False)  # [B, C]
            self._epoch_argmin_indices.setdefault(stage, []).append(argmin_indices.detach().cpu())

    def _build_code_metrics(self, stage: str, indices: torch.Tensor, metric_prefix: str = "") -> dict[str, float]:
        report = compute_sid_health_report(
            indices.numpy(),
            self.rqvae.rq.codebook_size_list,
            code_offset=0,
            top_k=0,
            include_histograms=False,
        )
        prefix = f"{stage}/{metric_prefix}"
        metrics: dict[str, float] = {
            f"{prefix}unique_code_path_ratio": float(report["paths"]["unique_code_path_ratio"]),
            f"{prefix}code_path_collision_rate": float(report["paths"]["collision_rate"]),
            f"{prefix}code_path_collisions": float(report["paths"]["num_collisions"]),
        }
        for level in report["levels"]:
            level_idx = level["level"]
            metrics[f"{prefix}code_usage_l{level_idx}"] = float(level["used_ratio"])
            metrics[f"{prefix}used_codes_l{level_idx}"] = float(level["used_codes"])
            metrics[f"{prefix}dead_codes_l{level_idx}"] = float(level["dead_codes"])
            metrics[f"{prefix}max_bucket_share_l{level_idx}"] = float(level["max_bucket_share"])
            metrics[f"{prefix}entropy_l{level_idx}"] = float(level["entropy"])
            metrics[f"{prefix}perplexity_l{level_idx}"] = float(level["perplexity"])
        return metrics

    def _log_epoch_code_metrics(self, stage: str) -> None:
        indices_list = self._epoch_indices.get(stage, [])
        if not indices_list:
            return
        indices = torch.cat(indices_list, dim=0)  # [N, C]
        metrics = self._build_code_metrics(stage, indices)

        argmin_indices_list = self._epoch_argmin_indices.get(stage, [])
        if argmin_indices_list:
            argmin_indices = torch.cat(argmin_indices_list, dim=0)  # [N, C]
            metrics.update(self._build_code_metrics(stage, argmin_indices, metric_prefix="argmin_"))
            comparison = compare_code_assignments(indices.numpy(), argmin_indices.numpy())
            metrics[f"{stage}/sk_argmin_path_agreement"] = float(comparison["exact_path_agreement_rate"])
            for level in comparison["per_level_agreement"]:
                metrics[f"{stage}/sk_argmin_agreement_l{level['level']}"] = float(level["agreement_rate"])

        self.log_dict(metrics, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = self._optimizer(self.rqvae.parameters())
        if self._scheduler is None:
            return optimizer
        scheduler = self._scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val/total_loss"},
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.rqvae(x, use_sk=self.use_sk)

    def _compute_loss(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        x_out, rq_loss, indices = self(batch)
        recon_loss = F.mse_loss(x_out, batch)
        total_loss = recon_loss + rq_loss
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "rq_loss": rq_loss,
            "indices": indices,
        }

    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        loss_dict = self._compute_loss(batch)
        if stage == "train" and self._pending_kmeans_metrics:
            self.log_dict(self._pending_kmeans_metrics, prog_bar=False, sync_dist=True)
            self._pending_kmeans_metrics = {}
        self.log(
            f"{stage}/total_loss", loss_dict["total_loss"], prog_bar=(stage != "test"), on_step=False, on_epoch=True
        )
        self.log(f"{stage}/recon_loss", loss_dict["recon_loss"], prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/rq_loss", loss_dict["rq_loss"], prog_bar=False, on_step=False, on_epoch=True)
        self._track_code_metrics(stage, batch, loss_dict["indices"])
        return loss_dict["total_loss"]

    def on_train_epoch_start(self) -> None:
        self._reset_code_metrics("train")

    def on_train_epoch_end(self) -> None:
        self._log_epoch_code_metrics("train")

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="val")

    def on_validation_epoch_start(self) -> None:
        self._reset_code_metrics("val")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_code_metrics("val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="test")

    def on_test_epoch_start(self) -> None:
        self._reset_code_metrics("test")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_code_metrics("test")
