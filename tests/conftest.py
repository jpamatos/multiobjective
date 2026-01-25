import pytest
import torch


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


@pytest.fixture
def mock_mnist(monkeypatch):
    monkeypatch.setattr(
        "multiobjective.data.mnist.MNIST",
        FakeMNIST,
    )
