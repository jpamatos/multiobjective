import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST


class MNISTLoader:
    def __init__(
        self,
        train_size: int = 7000,
        test_size: int = 3000,
        data_dir: str = "data",
        random_state: int = 1,
    ) -> None:
        self.train_size = train_size
        self.test_size = test_size
        self.data_dir = data_dir
        self.random_state = random_state

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        train_ds = MNIST(
            root=self.data_dir, train=True, download=True
        )
        test_ds = MNIST(
            root=self.data_dir, train=False, download=True
        )

        train_images = train_ds.data.numpy()
        train_labels = train_ds.targets.numpy()
        test_images = test_ds.data.numpy()
        test_labels = test_ds.targets.numpy()

        train_images, _, train_labels, _ = train_test_split(
            train_images,
            train_labels,
            train_size=self.train_size,
            stratify=train_labels,
            random_state=self.random_state,
        )

        test_images, _, test_labels, _ = train_test_split(
            test_images,
            test_labels,
            train_size=self.test_size,
            stratify=test_labels,
            random_state=self.random_state,
        )

        train_images = train_images.astype("float32") / 255.0
        test_images = test_images.astype("float32") / 255.0

        train_images = train_images.reshape((-1, 28, 28, 1))
        test_images = test_images.reshape((-1, 28, 28, 1))

        num_classes = 10
        train_labels = np.eye(num_classes)[train_labels]
        test_labels = np.eye(num_classes)[test_labels]

        return train_images, test_images, train_labels, test_labels
