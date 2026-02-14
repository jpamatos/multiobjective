from typing import Protocol

import torch


class Classifier(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def parameters(self) -> None:
        ...
