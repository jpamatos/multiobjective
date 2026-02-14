import torch


def test_cnn_forward(fake_data):
    from multiobjective.models.cnn import CNNModel

    X_train, y_train, _, _ = fake_data

    model = CNNModel(
        conv_layers=2,
        conv_neurons=32,
        dropout=0.1,
        dense_neurons=64,
        num_classes=10,
    )

    x = torch.tensor(X_train).permute(0, 3, 1, 2)
    out = model(x)

    assert out.shape == (20, 10)

def test_cnn_determinism(fake_data):
    import torch
    
    from multiobjective.models.cnn import CNNModel

    torch.manual_seed(42)

    X_train, _, _, _ = fake_data
    x = torch.tensor(X_train).permute(0, 3, 1, 2)

    model1 = CNNModel(
        conv_layers=2,
        conv_neurons=32,
        dropout=0.25,
        dense_neurons=64,
        num_classes=10,
    )
    model1.eval()

    torch.manual_seed(42)
    model2 = CNNModel(
        conv_layers=2,
        conv_neurons=32,
        dropout=0.25,
        dense_neurons=64,
        num_classes=10,
    )
    model2.eval()

    out1 = model1(x)
    out2 = model2(x)

    assert torch.allclose(out1, out2)
