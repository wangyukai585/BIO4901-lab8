import importlib.util, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
import numpy as np
import pandas as pd
import pytest
import torch
from utils import *
from train_core import next_fallback, checkpoint, restore

spec = importlib.util.spec_from_file_location(
    "evaluate", ROOT / "codes/06_evaluate_and_bootstrap.py"
)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_official_headers_and_exclusions(tmp_path):
    p = tmp_path / "x.fasta"
    p.write_text(
        ">a Cell.membrane-M test\nACDE\n>b Plastid-U\nACDF\n>c Cytoplasm-Nucleus-U\nAAAA\n"
    )
    df = parse_fasta(p)
    assert len(df) == 2
    assert df.iloc[0].split == "test"
    assert df.iloc[1].location == "Chloroplast"
    assert leakage_check(df)["cross_split_identical_sequence_pairs"] == 0
    df.loc[1, "sequence_hash"] = df.loc[0, "sequence_hash"]
    assert leakage_check(df)["cross_split_identical_sequence_pairs"] == 1


def test_sampling_never_changes_official_split():
    df = parse_fasta(ROOT / "data/raw/deeploc_data.fasta")
    a = cap_split(df[df.split == "test"], 40, 42)
    b = cap_split(df[df.split == "test"], 40, 42)
    assert a.equals(b)
    assert set(a.split) == {"test"}
    assert a.label.nunique() == 10
    assert not set(a.id) & set(df[df.split == "train"].id)


def test_trim_and_pooling_excludes_special_tokens():
    assert trim("ABCDEFGHIJ", 6) == "ABCHIJ"

    class Backbone:
        def __call__(self, **b):
            class O:
                pass

            o = O()
            o.last_hidden_state = torch.tensor([[[99.0], [2.0], [4.0], [99.0], [99.0]]])
            return o

    b = {
        "input_ids": torch.zeros((1, 5), dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "special_tokens_mask": torch.tensor([[1, 0, 0, 1, 1]]),
    }
    assert pooled(Backbone(), b).item() == 3


def test_calibration_boundaries():
    y = np.array([0, 1])
    p = np.eye(10)[y]
    ece, cal = ev.calibration(y, p)
    assert ece == 0
    assert cal.n.sum() == 2
    assert cal.iloc[-1].n == 2
    p = np.ones((2, 10)) / 10
    ece, _ = ev.calibration(y, p)
    assert ece == pytest.approx(0.4)


def test_paired_bootstrap_same_model_zero_difference():
    y = np.tile(np.arange(10), 3)
    p = np.eye(10)[y] * 0.9 + 0.01
    out = ev.bootstrap(y, {"a": p, "b": p}, 20, 1)
    diff = out[out.kind == "paired_difference"]
    assert np.all(diff[["estimate", "ci_low", "ci_high"]].to_numpy() == 0)


def test_fallback_order():
    a = argparse.Namespace(
        batch_size=4, max_length=512, min_length=128, freeze_layers=0
    )
    assert [next_fallback(a) for _ in range(5)] == [
        "reduce_batch",
        "reduce_batch",
        "reduce_length",
        "reduce_length",
        "freeze_early_layers",
    ]
    assert (a.batch_size, a.max_length, a.freeze_layers) == (1, 128, 5)


def test_checkpoint_roundtrip_optimizer(tmp_path):
    seed_all(42)
    m = torch.nn.Linear(3, 10)
    o = torch.optim.AdamW(m.parameters())
    x = torch.randn(2, 3)
    m(x).sum().backward()
    o.step()
    expected = m(x).detach().clone()
    path = tmp_path / "last.pt"
    a = argparse.Namespace(seed=42)
    checkpoint(path, m, o, 1, 1, a, [])
    with torch.no_grad():
        m.weight.zero_()
    restore(path, m, o)
    torch.testing.assert_close(expected, m(x))
    assert len(o.state) == 2


def test_knn_against_direct_reference():
    from frozen_knn import FrozenKNN

    x = np.eye(10)
    y = np.arange(10)
    clf = FrozenKNN(5).fit(x, y)
    np.testing.assert_allclose(clf.predict_proba(x), np.eye(10))
    p = clf.predict_proba(np.ones((1, 10)))
    assert p.shape == (1, 10)
    assert np.isclose(p.sum(), 1)


def test_probe_retries_oom_and_discards_probe_model(monkeypatch, tmp_path):
    import train_core as tc

    attempts = []

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(2, 10)

        def forward(self, b):
            return self.w(b)

    def build(a, stage):
        attempts.append(a.batch_size)
        if len(attempts) < 3:
            raise RuntimeError("CUDA out of memory (injected test)")
        return None, Tiny()

    monkeypatch.setattr(tc, "build_model", build)
    monkeypatch.setattr(
        tc, "tokenize", lambda tok, seqs, n, d: torch.ones((len(seqs), 2))
    )
    a = argparse.Namespace(
        seed=42,
        batch_size=4,
        max_length=128,
        min_length=128,
        freeze_layers=0,
        precision="fp32",
        optimizer="adamw",
        lr=0.001,
    )
    out = tc.probe_memory(a, torch.device("cpu"), logger(tmp_path))
    assert attempts == [4, 2, 1]
    assert [x["success"] for x in out] == [False, False, True]
