import torch
from torchvision.models import resnet18

from aoi.modeling import AnomalyTensorModel, build_tensor_model, validate_model_state


def _backbone_state() -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in resnet18(weights=None).state_dict().items()}


def _padim_state() -> dict:
    return {
        "model_name": "padim",
        "backbone": "resnet18",
        "layers": ("layer2",),
        "image_size": 32,
        "pre_trained": False,
        "feature_indices": torch.tensor([0, 1]),
        "backbone_state": _backbone_state(),
        "mean": torch.zeros(4, 4, 2),
        "inv_std": torch.ones(4, 4, 2),
    }


def test_padim_tensor_model_contract() -> None:
    model = build_tensor_model(_padim_state(), device="cpu")
    anomaly_map, score = model(torch.rand(2, 3, 32, 32))
    assert anomaly_map.shape == (2, 1, 32, 32)
    assert score.shape == (2,)
    assert torch.equal(score, anomaly_map.flatten(1).max(dim=1).values)


def test_patchcore_head_matches_squared_distance_reference() -> None:
    features = torch.tensor([[[[1.0]], [[0.0]]]])
    bank = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    distance = AnomalyTensorModel.patchcore_distance(features, bank)
    assert torch.allclose(distance, torch.zeros(1, 1, 1), atol=1e-6)


def test_patchcore_distance_is_continuous_near_zero() -> None:
    bank = torch.tensor([[1.0, 0.0]])
    low_angle = torch.tensor(0.0009)
    high_angle = torch.tensor(0.0011)
    low = torch.stack((torch.cos(low_angle), torch.sin(low_angle))).view(1, 2, 1, 1)
    high = torch.stack((torch.cos(high_angle), torch.sin(high_angle))).view(1, 2, 1, 1)

    low_distance = AnomalyTensorModel.patchcore_distance(low, bank)
    high_distance = AnomalyTensorModel.patchcore_distance(high, bank)

    assert float(high_distance - low_distance) < 1e-4


def test_model_state_rejects_non_finite_tensors() -> None:
    state = _padim_state()
    state["mean"][0, 0, 0] = float("nan")
    try:
        validate_model_state(state)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite model state was accepted")
