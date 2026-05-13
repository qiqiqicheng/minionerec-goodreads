import math
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class RQVAEDataset(Dataset):
    def __init__(self, embeddings: np.ndarray):
        super().__init__()
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings with shape [N, D], got {embeddings.shape}")
        self.embeddings = torch.from_numpy(embeddings).float()  # [N, D]

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.embeddings[index]  # [D]


class RQVAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        embedding_path: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        train_ratio: float = 0.9,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.0,
        seed: int = 728,
        persistent_workers: bool = False,
        embedding_dim: int | None = None,
    ):
        super().__init__()
        ratio_sum = train_ratio + valid_ratio + test_ratio
        if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

        self.embedding_path = Path(embedding_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.persistent_workers = persistent_workers and num_workers > 0
        self.embedding_dim = embedding_dim

        self.dataset: RQVAEDataset | None = None
        self.data_train: Subset | None = None
        self.data_valid: Subset | None = None
        self.data_test: Subset | None = None

    def prepare_data(self) -> None:
        if not self.embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")

    def setup(self, stage: str | None = None) -> None:
        _ = stage
        if self.dataset is not None:
            return

        embeddings = np.load(self.embedding_path)
        if self.embedding_dim is not None and embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: {embeddings.shape[1]} != configured RQ-VAE in_dim {self.embedding_dim}")
        self.dataset = RQVAEDataset(embeddings)

        num_items = len(self.dataset)
        indices = np.arange(num_items)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)

        train_end = int(num_items * self.train_ratio)
        valid_end = train_end + int(num_items * self.valid_ratio)

        train_indices = indices[:train_end].tolist()
        valid_indices = indices[train_end:valid_end].tolist()
        test_indices = indices[valid_end:].tolist()

        if len(train_indices) == 0:
            raise ValueError(f"Empty train split detected with num_items={num_items}")

        self.data_train = Subset(self.dataset, train_indices)
        self.data_valid = Subset(self.dataset, valid_indices)
        self.data_test = Subset(self.dataset, test_indices)

    def train_dataloader(self) -> DataLoader:
        if self.data_train is None:
            raise RuntimeError("Call setup() before requesting train_dataloader()")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        if self.data_valid is None:
            raise RuntimeError("Call setup() before requesting val_dataloader()")
        if len(self.data_valid) == 0:
            return []
        return DataLoader(
            dataset=self.data_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.data_test is None:
            raise RuntimeError("Call setup() before requesting test_dataloader()")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=False,
        )
