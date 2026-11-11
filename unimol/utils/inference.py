"""Device selection and tensor placement for single-process inference."""

from types import ModuleType

import torch
from unicore import utils


def add_device_args(parser):
    parser.add_argument(
        "--device", default="auto", metavar="DEVICE",
        help="auto (default), cpu, or a PyTorch device such as cuda:0, xpu:0, mps",
    )


def _device_modules():
    """Find installed device modules without importing optional vendor packages."""
    modules = {}
    for name, module in list(vars(torch).items()):
        if name.startswith("_") or not isinstance(module, ModuleType):
            continue
        try:
            device = torch.device(name)
        except (RuntimeError, ValueError, TypeError):
            continue
        if device.type in ("cpu", "meta", "lazy"):
            continue
        if callable(getattr(module, "is_available", None)):
            modules[device.type] = module

    # Older PyTorch releases expose the MPS probe only in torch.backends.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and callable(getattr(mps, "is_available", None)):
        modules.setdefault("mps", mps)
    return modules


def _available_device(device, modules, device_id):
    if device.type == "cpu":
        if device.index not in (None, 0):
            raise ValueError("CPU only supports device index 0")
        return torch.device("cpu")

    module = modules.get(device.type)
    if module is None:
        raise ValueError("no installed availability probe for this backend")
    if not module.is_available():
        raise ValueError("backend is not available in this PyTorch installation")

    index = device.index if device.index is not None else device_id
    count_devices = getattr(module, "device_count", None)
    if callable(count_devices):
        count = count_devices()
        if index < 0 or index >= count:
            raise ValueError("device index {} is out of range ({} devices)".format(index, count))
    elif index != 0:
        raise ValueError("backend does not expose device_count(); only index 0 can be selected")
    return torch.device(device.type, index)


def resolve_device(args):
    """Auto-detect an accelerator, or strictly honor an explicit device request."""
    requested = getattr(args, "device", "auto")
    if getattr(args, "cpu", False):
        if requested != "auto":
            try:
                is_cpu = torch.device(requested).type == "cpu"
            except (RuntimeError, ValueError, TypeError):
                is_cpu = False
            if not is_cpu:
                raise ValueError("--cpu cannot be combined with --device {}".format(requested))
        else:
            requested = "cpu"

    device_id = getattr(args, "device_id", 0)
    if requested == "auto":
        if device_id < 0:
            raise ValueError("--device-id must be non-negative")
        modules = _device_modules()
        priority = ("cuda", "xpu", "mps")
        candidates = list(priority) + sorted(set(modules) - set(priority))
        for name in candidates:
            if name not in modules:
                continue
            try:
                return _available_device(torch.device(name), modules, device_id)
            except (RuntimeError, ValueError, TypeError, AssertionError, ImportError, OSError):
                # A broken/unavailable backend must not prevent CPU fallback.
                continue
        return torch.device("cpu")

    try:
        device = torch.device(requested)
        # Explicit CPU inference does not need to inspect accelerator runtimes.
        modules = {} if device.type == "cpu" else _device_modules()
        return _available_device(device, modules, device_id)
    except (RuntimeError, ValueError, TypeError, AssertionError, ImportError, OSError) as exc:
        raise ValueError("Cannot use --device {}: {}".format(requested, exc)) from exc


def inference_device(args, logger):
    device = resolve_device(args)
    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 inference is supported only on CUDA/ROCm; omit --fp16 for FP32")
    if getattr(args, "bf16", False):
        raise ValueError("--bf16 inference is not supported; omit --bf16 for FP32")
    logger.info("Inference device: %s | precision: %s", device, "FP16" if args.fp16 else "FP32")
    return device


def move_to_device(sample, device):
    """Move tensors, preserving their dtype and all non-tensor sample metadata."""
    return utils.apply_to_sample(lambda tensor: tensor.to(device=device), sample)
