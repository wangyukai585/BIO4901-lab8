#!/usr/bin/env python
"""Post-training matched-length masking audit; never selects or updates models."""

from utils import *
from train_core import build_model, restore
from frozen_knn import FrozenKNN
from tqdm.auto import tqdm
import time


def masked(seq, condition, maximum=512):
    s = trim(seq, maximum)
    k = min(10, len(s) // 4)
    if condition == "original" or k == 0:
        return s
    starts = {"N10": 0, "C10": len(s) - k, "internal10": len(s) // 4 - k // 2}
    start = starts[condition]
    return s[:start] + "X" * k + s[start + k :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, default=ROOT / "results/full_run")
    p.add_argument("--data", type=Path, default=ROOT / "data/raw/deeploc_data.fasta")
    p.add_argument("--embedding_dir", type=Path, default=ROOT / "data/embeddings_full")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    a = p.parse_args()
    d = get_device(a.device)
    out = a.results_dir / "biological_controls"
    out.mkdir(exist_ok=True)
    log = logger(out)
    start = time.monotonic()
    data = parse_fasta(a.data).set_index("id")
    reference = pd.read_csv(a.results_dir / "full/predictions.csv")
    te = data.loc[reference.id].reset_index()
    if not (te.split == "test").all():
        raise ValueError("Non-test sample")
    log.info(
        "Post-training audit: %s test proteins, device=%s; initial estimate ~10 min on RTX 3080 Ti",
        len(te),
        d,
    )
    conditions = ["original", "N10", "C10", "internal10"]
    for stage in ["frozen", "lora", "full"]:
        run = json.loads((a.results_dir / stage / "run.json").read_text())
        if run["status"] != "FULL_RUN" or run.get("limited_steps"):
            raise ValueError("Full runs required")
        if run["dataset_sha256"] != data_digest(a.data):
            raise ValueError("Data changed")
        cfg = argparse.Namespace(**run["config"])
        cfg.gradient_checkpointing = False
        seed_all(cfg.seed)
        if stage == "frozen":
            tok, model = load_backbone(cfg)
            emb = np.load(a.embedding_dir / "embeddings.npz")
            knn = FrozenKNN().fit(emb["train"], emb["y_train"])
            head = torch.nn.Linear(640, 10).to(d)
            restore(a.results_dir / "linear/last.pt", head)
            head.eval()
        else:
            tok, model = build_model(cfg, stage)
            restore(a.results_dir / stage / "last.pt", model)
        model.to(d).eval()
        for condition in conditions:
            targets = ["frozen", "linear"] if stage == "frozen" else [stage]
            paths = [out / f"{target}_{condition}.csv" for target in targets]
            if all(x.exists() for x in paths):
                continue
            chunks = []
            with torch.inference_mode():
                for pos in tqdm(
                    range(0, len(te), a.batch_size), desc=f"{stage}/{condition}"
                ):
                    seqs = [
                        masked(s, condition, cfg.max_length)
                        for s in te.sequence.iloc[pos : pos + a.batch_size]
                    ]
                    b = tokenize(tok, seqs, cfg.max_length, d)
                    with amp(d, precision(d, "auto")):
                        y = pooled(model, b) if stage == "frozen" else model(b)
                    chunks.append(y.float().cpu().numpy())
                    if pos % (a.batch_size * 50) == 0:
                        log.info(
                            "%s %s %d/%d elapsed %.1fs",
                            stage,
                            condition,
                            pos,
                            len(te),
                            time.monotonic() - start,
                        )
            y = np.concatenate(chunks)
            if stage == "frozen":
                save_predictions(paths[0], te, knn.predict_proba(y))
                with torch.inference_mode(), amp(d, precision(d, "auto")):
                    prob = (
                        head(torch.tensor(y, device=d))
                        .float()
                        .softmax(-1)
                        .cpu()
                        .numpy()
                    )
                save_predictions(paths[1], te, prob)
            else:
                save_predictions(paths[0], te, torch.tensor(y).softmax(-1).numpy())
        del model
        clear_device(d)
    rows = []
    rng = np.random.default_rng(42)
    for stage in ["frozen", "linear", "lora", "full"]:
        baseline = pd.read_csv(out / f"{stage}_original.csv")
        published = pd.read_csv(a.results_dir / stage / "predictions.csv")
        assert baseline.id.tolist() == published.id.tolist()
        agreement = float((baseline.prediction == published.prediction).mean())
        if agreement < 0.99:
            raise ValueError(
                f"{stage}: restored baseline disagrees with published predictions"
            )
        write_json(
            out / f"{stage}_baseline_check.json",
            {
                "prediction_agreement": agreement,
                "note": "Supplement uses bf16 and batch 8 by default, while original frozen embeddings used fp32. All masks are paired with the rerun original baseline; main benchmark unchanged.",
            },
        )
        for condition in conditions[1:]:
            pert = pd.read_csv(out / f"{stage}_{condition}.csv")
            assert pert.id.tolist() == baseline.id.tolist()
            for label in range(10):
                ix = baseline.label.to_numpy() == label
                before = (baseline.prediction.to_numpy()[ix] == label).astype(float)
                after = (pert.prediction.to_numpy()[ix] == label).astype(float)
                delta = before - after
                boot = delta[rng.integers(0, len(delta), size=(1000, len(delta)))].mean(
                    1
                )
                rows.append(
                    dict(
                        strategy=stage,
                        condition=condition,
                        label=label,
                        category=CLASSES[label],
                        n=int(ix.sum()),
                        baseline_recall=before.mean(),
                        masked_recall=after.mean(),
                        recall_drop=delta.mean(),
                        ci_low=np.quantile(boot, 0.025),
                        ci_high=np.quantile(boot, 0.975),
                        true_probability_drop=(
                            baseline[f"p{label}"][ix] - pert[f"p{label}"][ix]
                        ).mean(),
                    )
                )
    pd.DataFrame(rows).to_csv(out / "class_effects.csv", index=False)
    write_json(
        out / "audit.json",
        dict(
            status="COMPLETED",
            test_n=len(te),
            conditions=conditions,
            max_residues=512,
            mask_length="min(10, floor(length/4))",
            internal_start="floor(retained_length/4)-floor(mask_length/2)",
            bootstrap="1000 paired within-class sequence resamples; descriptive unadjusted intervals",
            training=False,
            elapsed_seconds=time.monotonic() - start,
            interpretation="Post hoc diagnostic after primary benchmark; X masks are not biological deletions or causal validation.",
            device=str(d),
            precision="bf16 on CUDA when supported; same for all paired conditions",
            batch_size=a.batch_size,
        ),
    )


if __name__ == "__main__":
    main()
