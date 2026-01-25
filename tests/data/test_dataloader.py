import numpy as np


def test_load_returns_four_arrays(mock_mnist):
    from multiobjective.data.mnist import MNISTLoader

    loader = MNISTLoader(train_size=50, test_size=30)
    result = loader.load()

    assert isinstance(result, tuple)
    assert len(result) == 4


def test_shapes_are_correct(mock_mnist):
    from multiobjective.data.mnist import MNISTLoader

    loader = MNISTLoader(train_size=50, test_size=30)
    X_train, X_test, y_train, y_test = loader.load()

    assert X_train.shape == (50, 28, 28, 1)
    assert X_test.shape == (30, 28, 28, 1)
    assert y_train.shape == (50, 10)
    assert y_test.shape == (30, 10)


def test_images_are_normalized(mock_mnist):
    from multiobjective.data.mnist import MNISTLoader

    loader = MNISTLoader(train_size=50, test_size=30)
    X_train, X_test, _, _ = loader.load()

    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0
    assert X_test.min() >= 0.0
    assert X_test.max() <= 1.0


def test_labels_are_one_hot(mock_mnist):
    from multiobjective.data.mnist import MNISTLoader

    loader = MNISTLoader(train_size=50, test_size=30)
    _, _, y_train, y_test = loader.load()

    assert np.all(y_train.sum(axis=1) == 1)
    assert np.all(y_test.sum(axis=1) == 1)