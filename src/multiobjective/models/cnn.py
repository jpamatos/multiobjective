import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(
        self,
        conv_layers: int,
        conv_neurons: int,
        dropout: float,
        dense_neurons: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        layers = []
        in_channels = 1

        for _ in range(conv_layers):
            layers.extend(
                [
                    nn.Conv2d(in_channels, conv_neurons, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = conv_neurons

        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            out = self.conv(dummy)
            flattened = out.reshape(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened, dense_neurons),
            nn.ReLU(),
            nn.Linear(dense_neurons, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
