from typing import Callable, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from minionerec_goodreads.basic.activation import activation_layer


class MLP(nn.Module):
    def __init__(self, layers: list[int], dropout: float, activation: str = "relu", bn: bool = False):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.bn = bn

        mlps = []
        for i, (input, output) in enumerate(zip(layers[:-1], layers[1:])):
            mlps.append(nn.Dropout(self.dropout))
            mlps.append(nn.Linear(input, output))
            if self.bn and i != len(layers) - 2:  # exclude the last layer
                mlps.append(nn.BatchNorm1d(output))

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
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.size = size
        self.loss_beta = loss_beta
        self.embeddings = nn.Embedding(self.size, self.emb_dim)
        self.register_buffer("initted", torch.tensor(not kmeans_init))
        self.sk_temp = sk_temp
        self.sk_iters = sk_iters

        if not kmeans_init:
            self.embeddings.weight.data.uniform_(-1.0 / self.size, 1.0 / self.size)
        else:
            assert kmeans_iters > 0
            self.kmeans_iters = kmeans_iters

    def _kmeans_init(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): [B, D_emb]
        """
        _, _ = x.shape
        device = x.device
        x_np = x.detach().cpu().numpy()
        cluster = KMeans(n_clusters=self.size, max_iter=self.kmeans_iters).fit(x_np)
        centers = cluster.cluster_centers_  # [N, D_emb]
        self.embeddings.weight.data.copy_(torch.from_numpy(centers).to(device))
        self.initted.fill_(True)  # type: ignore

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

    def forward(self, x: torch.Tensor, use_sk: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [B, D_emb]
            use_sk (bool, optional): _description_. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            )
            for size in self.codebook_size_list
        ])

    def get_codebooks(self) -> list[torch.Tensor]:
        codebooks = []
        for cb in self.codebooks:
            codebooks.append(cb.codebook)  # [N_i, D_emb]
        return codebooks

    def forward(self, x: torch.Tensor, use_sk: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [B, D_emb]
            use_sk (bool): Whether to use the Sinkhorn-Knopp algorithm for soft assignment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return x_q_out, mean_losses, all_indices  # type: ignore


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

    def forward(self, x: torch.Tensor, use_sk: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): [B, D_in]
            use_sk (bool): Whether to use the Sinkhorn-Knopp algorithm for soft assignment.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_out (torch.Tensor): [B, D_in]
                - rq_loss (torch.Tensor): [,]
                - indices (torch.Tensor): [B, C]
        """
        x_encoded = self.encoder(x)  # [B, D_emb]
        x_q, rq_loss, indices = self.rq(x_encoded, use_sk)  # [B, D_emb], [,], [B, C]
        x_out = self.decoder(x_q)

        return x_out, rq_loss, indices

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
    ):
        super().__init__()
        self.rqvae = rqvae
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.use_sk = use_sk
        self.save_hyperparameters(logger=False, ignore=["rqvae"])

    def configure_optimizers(self):
        optimizer = self._optimizer(self.rqvae.parameters())
        if self._scheduler is None:
            return optimizer
        scheduler = self._scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "monitor": "val/total_loss"},
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.log(
            f"{stage}/total_loss", loss_dict["total_loss"], prog_bar=(stage != "test"), on_step=False, on_epoch=True
        )
        self.log(f"{stage}/recon_loss", loss_dict["recon_loss"], prog_bar=False, on_step=False, on_epoch=True)
        self.log(f"{stage}/rq_loss", loss_dict["rq_loss"], prog_bar=False, on_step=False, on_epoch=True)
        return loss_dict["total_loss"]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _ = batch_idx
        return self._shared_step(batch, stage="test")
