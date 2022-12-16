import tensorflow as tf
import torch

from nebullvm.tools.data import DataManager


def test_custom_input_data():
    input_data = [
        ((torch.randn(2, 3, 10, 10),), torch.randn(2, 1)),
        ((torch.randn(2, 3, 10, 10),), torch.randn(2, 1)),
        ((torch.randn(2, 3, 10, 10),), torch.randn(2, 1)),
        ((torch.randn(2, 3, 10, 10),), torch.randn(2, 1)),
    ]

    data_manager = DataManager(input_data)

    assert len(data_manager) == 4
    assert len(data_manager[0]) == 2
    assert len(data_manager[0][0]) == 1
    assert data_manager[0][0][0].shape == (2, 3, 10, 10)
    assert data_manager[0][1].shape == (2, 1)


def test_torch_dataloader_single_input_with_label():
    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3, 10, 10), torch.randn(8, 1)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    data_manager = DataManager.from_dataloader(dataloader)

    assert len(data_manager) == 4
    assert len(data_manager[0]) == 2
    assert len(data_manager[0][0]) == 1
    assert data_manager[0][0][0].shape == (2, 3, 10, 10)
    assert data_manager[0][1].shape == (2, 1)


def test_torch_dataloader_two_inputs_with_label():
    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3, 10, 10), torch.randn(8, 3, 10, 10), torch.randn(8, 1)
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    data_manager = DataManager.from_dataloader(dataloader)

    assert len(data_manager) == 4
    assert len(data_manager[0]) == 2
    assert len(data_manager[0][0]) == 2
    assert data_manager[0][0][0].shape == (2, 3, 10, 10)
    assert data_manager[0][0][1].shape == (2, 3, 10, 10)
    assert data_manager[0][1].shape == (2, 1)


def test_torch_dataloader_three_inputs_with_label():
    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3, 10, 10),
        torch.randn(8, 3, 10, 10),
        torch.randn(8, 3, 10, 10),
        torch.randn(8, 1),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    data_manager = DataManager.from_dataloader(dataloader)

    assert len(data_manager) == 4
    assert len(data_manager[0]) == 2
    assert len(data_manager[0][0]) == 3
    assert data_manager[0][0][0].shape == (2, 3, 10, 10)
    assert data_manager[0][0][1].shape == (2, 3, 10, 10)
    assert data_manager[0][0][2].shape == (2, 3, 10, 10)
    assert data_manager[0][1].shape == (2, 1)


def test_torch_dataloader_single_input_without_label():
    dataset = torch.utils.data.TensorDataset(torch.randn(8, 3, 10, 10))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    data_manager = DataManager.from_dataloader(dataloader)

    assert len(data_manager) == 4
    assert len(data_manager[0]) == 2
    assert len(data_manager[0][0]) == 1
    assert data_manager[0][0][0].shape == (2, 3, 10, 10)


def test_tensorflow_dataloader_single_input_with_label():
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([8, 10, 10, 3]), tf.random.normal([8, 1]))
    )
    data_manager = DataManager.from_dataloader(dataset.batch(2))

    assert len(data_manager) == 4
    assert len(data_manager[0]) == 2
    assert len(data_manager[0][0]) == 1
    assert data_manager[0][0][0].shape == (2, 10, 10, 3)
    assert data_manager[0][1].shape == (2, 1)


def test_tensorflow_dataloader_two_inputs_with_label():
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.random.normal([8, 10, 10, 3]),
            tf.random.normal([8, 10, 10, 3]),
            tf.random.normal([8, 1]),
        )
    )
    data_manager = DataManager.from_dataloader(dataset.batch(2))

    assert len(data_manager) == 4
    assert len(data_manager[0]) == 2
    assert len(data_manager[0][0]) == 2
    assert data_manager[0][0][0].shape == (2, 10, 10, 3)
    assert data_manager[0][0][1].shape == (2, 10, 10, 3)
    assert data_manager[0][1].shape == (2, 1)


def test_tensorflow_dataloader_three_inputs_with_label():
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.random.normal([8, 10, 10, 3]),
            tf.random.normal([8, 10, 10, 3]),
            tf.random.normal([8, 10, 10, 3]),
            tf.random.normal([8, 1]),
        )
    )
    data_manager = DataManager.from_dataloader(dataset.batch(2))

    assert len(data_manager) == 4
    assert len(data_manager[0]) == 2
    assert len(data_manager[0][0]) == 3
    assert data_manager[0][0][0].shape == (2, 10, 10, 3)
    assert data_manager[0][0][1].shape == (2, 10, 10, 3)
    assert data_manager[0][0][2].shape == (2, 10, 10, 3)
    assert data_manager[0][1].shape == (2, 1)


def test_tensorflow_dataloader_single_input_without_label():
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.random.normal([8, 10, 10, 3])
    )
    data_manager = DataManager.from_dataloader(dataset.batch(2))

    assert len(data_manager) == 4
    assert len(data_manager[0]) == 2
    assert len(data_manager[0][0]) == 1
    assert data_manager[0][0][0].shape == (2, 10, 10, 3)
