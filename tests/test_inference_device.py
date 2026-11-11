import argparse
import logging
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from unimol.utils import inference


def args(**overrides):
    values = dict(device="auto", cpu=False, device_id=0, fp16=False, bf16=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def backend(available=True, count=2):
    return SimpleNamespace(is_available=Mock(return_value=available),
                           device_count=Mock(return_value=count))


@pytest.mark.parametrize("available,expected", [
    ((True, True, True), "cuda:0"),
    ((False, True, True), "xpu:0"),
    ((False, False, True), "mps:0"),
    ((False, False, False), "cpu"),
])
def test_auto_priority(monkeypatch, available, expected):
    modules = {name: backend(flag) for name, flag in zip(("cuda", "xpu", "mps"), available)}
    monkeypatch.setattr(inference, "_device_modules", lambda: modules)
    assert str(inference.resolve_device(args())) == expected
    if available[0]:
        modules["xpu"].is_available.assert_not_called()
        modules["mps"].is_available.assert_not_called()


def test_auto_other_backends_are_alphabetical(monkeypatch):
    monkeypatch.setattr(inference, "_device_modules", lambda: {"mtia": backend(), "hpu": backend()})
    assert str(inference.resolve_device(args())) == "hpu:0"


@pytest.mark.parametrize("error", [RuntimeError, ValueError, TypeError, AssertionError, ImportError, OSError])
def test_auto_broken_runtime_and_invalid_index_are_skipped(monkeypatch, error):
    broken = backend()
    broken.is_available.side_effect = error("missing driver")
    monkeypatch.setattr(inference, "_device_modules", lambda: {"cuda": broken, "xpu": backend(count=1)})
    assert str(inference.resolve_device(args(device_id=3))) == "cpu"


def test_explicit_cpu_does_not_probe_accelerators(monkeypatch):
    probe = Mock(side_effect=AssertionError("must not probe"))
    monkeypatch.setattr(inference, "_device_modules", probe)
    assert inference.resolve_device(args(device="cpu")) == torch.device("cpu")
    assert inference.resolve_device(args(cpu=True)) == torch.device("cpu")


@pytest.mark.parametrize("device,cpu,device_id,expected", [
    ("cuda", False, 1, "cuda:1"),
    ("cuda:0", False, 1, "cuda:0"),
    ("cuda:1", False, -1, "cuda:1"),
    ("cpu", True, 0, "cpu"),
    ("cpu:0", True, 1, "cpu"),
])
def test_explicit_device_and_legacy_arguments(monkeypatch, device, cpu, device_id, expected):
    monkeypatch.setattr(inference, "_device_modules", lambda: {"cuda": backend()})
    assert str(inference.resolve_device(args(device=device, cpu=cpu, device_id=device_id))) == expected


@pytest.mark.parametrize("overrides,match", [
    ({"device": "cuda", "cpu": True}, "--cpu cannot"),
    ({"device": "cuda:2"}, "out of range"),
    ({"device": "cuda", "device_id": -1}, "out of range"),
    ({"device_id": -1}, "non-negative"),
    ({"device": "cpu:1"}, "CPU only"),
    ({"device": "not-a-backend"}, "Cannot use --device"),
    ({"device": "meta"}, "no installed availability probe"),
    ({"device": "lazy"}, "no installed availability probe"),
])
def test_invalid_requests_fail(monkeypatch, overrides, match):
    monkeypatch.setattr(inference, "_device_modules", lambda: {"cuda": backend()})
    with pytest.raises(ValueError, match=match):
        inference.resolve_device(args(**overrides))


def test_explicit_unavailable_device_never_falls_back(monkeypatch):
    monkeypatch.setattr(inference, "_device_modules", lambda: {"cuda": backend(False), "xpu": backend()})
    with pytest.raises(ValueError, match="not available"):
        inference.resolve_device(args(device="cuda"))


def test_mps_probe_without_device_count(monkeypatch):
    monkeypatch.setattr(inference, "_device_modules", lambda: {"mps": SimpleNamespace(is_available=lambda: True)})
    assert str(inference.resolve_device(args(device="mps"))) == "mps:0"
    with pytest.raises(ValueError, match="only index 0"):
        inference.resolve_device(args(device="mps:1"))


def test_discovery_includes_registered_backends(monkeypatch):
    module = ModuleType("torch.privateuseone")
    module.is_available = lambda: True
    module.device_count = lambda: 1
    monkeypatch.setattr(torch, "privateuseone", module, raising=False)
    modules = inference._device_modules()
    assert modules["privateuseone"] is module
    assert "cpu" not in modules
    assert "meta" not in modules
    assert "lazy" not in modules
    assert str(inference.resolve_device(args(device="privateuseone"))) == "privateuseone:0"


def test_parser_default_and_unrestricted_device_argument():
    parser = argparse.ArgumentParser()
    inference.add_device_args(parser)
    assert parser.parse_args([]).device == "auto"
    assert parser.parse_args(["--device", "privateuseone:0"]).device == "privateuseone:0"


def test_nested_sample_transfer_preserves_dtypes_and_metadata():
    sample = {"net_input": {"tokens": torch.tensor([1, 2]), "distance": torch.ones(2)},
              "names": ["CCO", "N"], "tuple": (torch.tensor([True]), None), "value": 7}
    # Meta makes a real device transfer test possible on CPU-only installations.
    result = inference.move_to_device(sample, torch.device("meta"))
    assert result["net_input"]["tokens"].device.type == "meta"
    assert result["net_input"]["tokens"].dtype == torch.int64
    assert result["net_input"]["distance"].dtype == torch.float32
    assert result["tuple"][0].dtype == torch.bool
    assert result["tuple"][1] is None
    assert result["names"] == sample["names"]
    assert result["value"] == 7
    assert sample["net_input"]["tokens"].device.type == "cpu"


@pytest.mark.parametrize("flag", ["fp16", "bf16"])
def test_unsupported_precision_fails(flag):
    with pytest.raises(ValueError, match="--" + flag):
        inference.inference_device(args(device="cpu", **{flag: True}), logging.getLogger(__name__))


def test_cuda_fp16_and_device_logging(monkeypatch, caplog):
    monkeypatch.setattr(inference, "_device_modules", lambda: {"cuda": backend()})
    with caplog.at_level(logging.INFO):
        device = inference.inference_device(args(device="cuda:1", fp16=True), logging.getLogger(__name__))
    assert str(device) == "cuda:1"
    assert "cuda:1 | precision: FP16" in caplog.text
