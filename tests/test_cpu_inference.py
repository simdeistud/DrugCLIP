"""Exercise real DrugCLIP encoders and LMDB pipelines with a small CPU model."""

from argparse import Namespace
from pathlib import Path
import pickle
import sys
from unittest.mock import Mock

import lmdb
import numpy as np
import pytest
import torch

from unicore import checkpoint_utils, tasks
from unimol import retrieval, test as benchmark


ROOT = Path(__file__).resolve().parents[1]


def write_lmdb(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with lmdb.open(str(path), subdir=False, map_size=4 * 1024 * 1024) as env:
        with env.begin(write=True) as txn:
            for i, entry in enumerate(entries):
                txn.put(str(i).encode(), pickle.dumps(entry))


@pytest.fixture
def cpu_fixture(tmp_path, monkeypatch):
    torch.manual_seed(1)
    monkeypatch.chdir(tmp_path)
    settings = Namespace(
        data=str(ROOT / "data"), task="drugclip", arch="drugclip", seed=1,
        device="cpu", cpu=False, device_id=0, fp16=False, bf16=False,
        max_seq_len=512, max_pocket_atoms=256,
        finetune_mol_model=None, finetune_pocket_model=None,
        mol_encoder_layers=1, mol_encoder_embed_dim=16,
        mol_encoder_ffn_embed_dim=32, mol_encoder_attention_heads=4,
        pocket_encoder_layers=1, pocket_encoder_embed_dim=16,
        pocket_encoder_ffn_embed_dim=32, pocket_encoder_attention_heads=4,
        mol_path=str(tmp_path / "mols.lmdb"), pocket_path=str(tmp_path / "pocket.lmdb"),
        emb_dir=str(tmp_path / "emb"), path=str(tmp_path / "checkpoint.pt"),
    )
    mols = [dict(atoms=["C", "N", "O"],
                 coordinates=[np.array([[0, 0, 0], [1 + i / 100, 0, 0], [0, 1, 0]], dtype=np.float32)],
                 smi="compound_{}".format(i), label=int(i % 10 == 0)) for i in range(220)]
    pockets = [dict(pocket_atoms=["C", "N", "O", "S"],
                    pocket_coordinates=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
                    pocket="pocket_0")]
    write_lmdb(Path(settings.mol_path), mols)
    write_lmdb(Path(settings.pocket_path), pockets)
    for directory, filename in (("DUD-E/raw/all/fixture", "pocket.lmdb"),
                                ("lit_pcba/fixture", "pockets.lmdb")):
        write_lmdb(tmp_path / "data" / directory / "mols.lmdb", mols)
        write_lmdb(tmp_path / "data" / directory / filename, pockets)
    task = tasks.setup_task(settings)
    model = task.build_model(settings).float().eval()
    torch.save({"model": model.state_dict()}, settings.path)
    return settings, task, model


def test_cpu_checkpoint_and_retrieval_cache(cpu_fixture, monkeypatch):
    settings, task, model = cpu_fixture
    state = checkpoint_utils.load_checkpoint_to_cpu(settings.path)
    model.load_state_dict(state["model"], strict=True)
    assert model.logit_scale.shape == (1,)
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    np.testing.assert_allclose(model.logit_scale.detach().numpy(), [np.log(14)], rtol=1e-6)

    def assert_no_grad(module, inputs):
        assert not torch.is_grad_enabled()

    model.mol_model.encoder.register_forward_pre_hook(assert_no_grad)
    model.pocket_model.encoder.register_forward_pre_hook(assert_no_grad)
    names, scores = task.retrieve_mols(model, settings.mol_path, settings.pocket_path, settings.emb_dir, 10)
    assert len(names) == len(scores) == 10
    assert np.isfinite(scores).all()
    assert np.all(scores[:-1] >= scores[1:])
    monkeypatch.setattr(task, "load_retrieval_mols_dataset", lambda *a: pytest.fail("cache was not reused"))
    # Reuse the cache directly: UniCore retains LMDB handles in global LRU caches,
    # so reopening the same database in this process is not supported by py-lmdb.
    embeddings, cached_names = task.encode_mols_once(
        model, settings.mol_path, settings.emb_dir, "atoms", "coordinates")
    assert len(cached_names) == len(embeddings) == 220
    assert set(names).issubset(cached_names)
    assert np.isfinite(embeddings).all()


@pytest.mark.parametrize("test_task", ["DUDE", "PCBA"])
def test_cpu_benchmark_entrypoints(cpu_fixture, test_task, monkeypatch):
    settings, task, model = cpu_fixture
    settings.test_task = test_task
    method_name = "test_dude_target" if test_task == "DUDE" else "test_pcba_target"
    original = getattr(task, method_name)
    results = []

    def record_metrics(*args, **kwargs):
        metrics = original(*args, **kwargs)
        results.append(metrics)
        return metrics

    monkeypatch.setattr(task, method_name, record_metrics)
    monkeypatch.setattr(benchmark.tasks, "setup_task", lambda _: task)
    benchmark.main(settings)
    assert len(results) == 1
    assert np.isfinite(results[0][0])
    assert np.isfinite(results[0][1])


def test_cpu_retrieval_entrypoint(cpu_fixture):
    settings, _, _ = cpu_fixture
    retrieval.main(settings)
    lines = (Path(settings.emb_dir) / "ranked_compounds.txt").read_text().splitlines()
    assert len(lines) == 220
    assert all(len(line.split("\t")) == 2 for line in lines)


@pytest.mark.parametrize("entrypoint", [retrieval, benchmark])
def test_cli_rejects_unavailable_device_before_checkpoint_loading(monkeypatch, entrypoint):
    from unimol.utils import inference

    monkeypatch.setattr(inference, "_device_modules", lambda: {})
    loader = Mock(side_effect=AssertionError("checkpoint must not be opened"))
    monkeypatch.setattr(checkpoint_utils, "load_checkpoint_to_cpu", loader)
    # unimol is already registered by this test module's imports.
    monkeypatch.setattr(sys, "argv", [entrypoint.__file__, str(ROOT / "data"),
        "--task", "drugclip", "--loss", "in_batch_softmax",
        "--arch", "drugclip", "--device", "cuda", "--path", "never-opened.pt"])
    with pytest.raises(ValueError, match="Cannot use --device cuda"):
        entrypoint.cli_main()
    loader.assert_not_called()


def test_auto_does_not_retry_after_inference_failure(monkeypatch):
    from unimol.utils import inference

    monkeypatch.setattr(inference, "_device_modules", lambda: {})
    model = torch.nn.Linear(1, 1)
    task = Mock()
    task.build_model.return_value = model
    task.retrieve_mols.side_effect = RuntimeError("operator is not implemented")
    monkeypatch.setattr(tasks, "setup_task", lambda _: task)
    monkeypatch.setattr(checkpoint_utils, "load_checkpoint_to_cpu", lambda _: {"model": model.state_dict()})
    settings = Namespace(device="auto", cpu=False, device_id=0, fp16=False, bf16=False,
                         path="unused.pt", mol_path="unused.lmdb", pocket_path="unused.lmdb", emb_dir="unused")
    with pytest.raises(RuntimeError, match="operator is not implemented"):
        retrieval.main(settings)
    task.retrieve_mols.assert_called_once()
