import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


class FakeMNIST:
    def __init__(self, *args, **kwargs):
        num_classes = 10
        samples_per_class = 20
        labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
        images = torch.randint(
        0, 256,
        (num_classes * samples_per_class, 28, 28),
        dtype=torch.uint8,
        )
        self.data = images
        self.targets = labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def mock_mnist(monkeypatch):
    monkeypatch.setattr(
        "multiobjective.data.mnist.MNIST",
        FakeMNIST,
    )

@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "genome_cfg": {
                "size": 8,
                "slices": {
                    "conv_layers": [0, 1],
                    "neurons_conv": [1, 4],
                    "dropout": [4, 6],
                    "neurons_dense": [6, 8],
                },
                "mappings": {
                    "conv_layers_genes": {"0": 2, "1": 3},
                    "neurons_conv_genes": {
                        "000": 32,
                        "001": 64,
                        "010": 96,
                        "011": 128,
                        "100": 192,
                        "101": 256,
                        "110": 384,
                        "111": 512,
                    },
                    "dropout_genes": {
                        "00": 0.0,
                        "01": 0.1,
                        "10": 0.25,
                        "11": 0.4,
                    },
                    "neurons_dense_genes": {
                        "00": 32,
                        "01": 64,
                        "10": 128,
                        "11": 256,
                    },
                },
            },
           "model": {
                "_target_": "multiobjective.models.cnn.CNNModel",
                "num_classes": 10,
            }
        }
    )

@pytest.fixture
def fake_data():
    X_train = np.random.rand(20, 28, 28, 1).astype(np.float32)
    X_test = np.random.rand(10, 28, 28, 1).astype(np.float32)

    y_train = np.eye(10)[np.random.randint(0, 10, 20)]
    y_test = np.eye(10)[np.random.randint(0, 10, 10)]

    return X_train, X_test, y_train, y_test
