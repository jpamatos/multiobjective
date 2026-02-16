from time import time

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from multiobjective.protocols.model import Classifier
from multiobjective.utils.decoder import GenomeDecoder


class Individual:
    model: nn.Module
    hyperparams: dict | None

    def __init__(
        self,
        cfg: DictConfig,
        generation: int = 0,
        genome: list[int] | None = None,
    ) -> None:
        self.cfg = cfg
        self.generation = generation

        self.genome_size: int = cfg.genome_cfg.size
        self.genome: list[int] = (
            genome
            if genome is not None
            else np.random.randint(0, 2, self.genome_size).tolist()
        )
        self.metrics = {}

    def _set_seed(self) -> None:
        seed = int("".join(map(str, self.genome)), 2)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _decode_genome(self) -> dict:
        decoder = GenomeDecoder(self.genome, self.cfg.genome_cfg)

        return {
            "conv_layers": decoder.decode("conv_layers"),
            "conv_neurons": decoder.decode("neurons_conv"),
            "dropout": decoder.decode("dropout"),
            "dense_neurons": decoder.decode("neurons_dense"),
            "num_classes": self.cfg.model.num_classes,
        }


    def _build_model(self) -> nn.Module:
        self.hyperparams = self._decode_genome()

        return hydra.utils.instantiate(
            self.cfg.model,
            **self.hyperparams,
        )

    def evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        *,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_seed()

        model = self._build_model().to(device)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        X_train_t = torch.tensor(X_train).permute(0, 3, 1, 2).float()
        X_test_t = torch.tensor(X_test).permute(0, 3, 1, 2).float()

        y_train_t = torch.tensor(np.argmax(y_train, axis=1)).long()
        y_test_t = torch.tensor(np.argmax(y_test, axis=1)).long()

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=True,
        )

        # -------- training --------
        model.train()
        loss = torch.tensor(0.0)
        for _ in range(epochs):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        # -------- evaluation --------
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

        y_true = y_test_t.numpy()

        accuracy = (preds == y_true).mean()
        f1 = f1_score(y_true, preds, average="macro")

        try:
            auc = roc_auc_score(y_true, probs, multi_class="ovr")
        except ValueError:
            auc = 0.0

        weights_norm = sum(p.norm().item() for p in model.parameters())

        start = time()
        _ = model(X_test_t.to(device))
        latency = (time() - start) / len(X_test_t)

        self.metrics = {
            "loss": (1, loss.item()),
            "accuracy": (-1, accuracy),
            "f1_score": (-1, f1),
            "auc": (-1, auc),
            "weights_norm": (1, weights_norm),
            "latency": (1, latency),
        }

        self.model = model

    def crossover(self, other: "Individual") -> list["Individual"]:
        cut = np.random.randint(1, self.genome_size - 1)

        return [
            Individual(self.cfg, self.generation + 1, self.genome[:cut] + other.genome[cut:]),
            Individual(self.cfg, self.generation + 1, other.genome[:cut] + self.genome[cut:]),
        ]

    def mutation(self, rate: float) -> "Individual":
        for i in range(self.genome_size):
            if np.random.rand() < rate:
                self.genome[i] ^= 1
        return self

    def __repr__(self) -> str:
        return f"Gen {self.generation} | genome={self.genome}"
