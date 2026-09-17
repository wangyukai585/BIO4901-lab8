#!/usr/bin/env python
"""Aligned heldout metrics; paired, stratified sequence bootstrap and robustness."""

from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from utils import *

STAGES = ["frozen", "linear", "lora", "full"]


def calibration(y, p, bins=10):
    confidence = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for k in range(bins):
        idx = bucket == k
        rows.append(
            dict(
                bin=k,
                n=int(idx.sum()),
                confidence=float(confidence[idx].mean()) if idx.any() else np.nan,
                accuracy=float(correct[idx].mean()) if idx.any() else np.nan,
            )
        )
    frame = pd.DataFrame(rows)
    valid = frame.n > 0
    ece = float(
        (
            frame.loc[valid, "n"]
            / len(y)
            * (frame.loc[valid, "accuracy"] - frame.loc[valid, "confidence"]).abs()
        ).sum()
    )
    return ece, frame


def metrics(y, p):
    if (
        not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1, atol=1e-4)
    ):
        raise ValueError("Invalid probabilities")
    p = p.astype(np.float64)
    p = p / p.sum(1, keepdims=True)  # CSV roundoff only.
    ece, _ = calibration(y, p)
    return dict(
        accuracy=accuracy_score(y, p.argmax(1)),
        macro_f1=f1_score(
            y, p.argmax(1), labels=np.arange(10), average="macro", zero_division=0
        ),
        ece=ece,
        nll=log_loss(y, p, labels=np.arange(10)),
        brier=float(np.mean(np.sum((p - np.eye(10)[y]) ** 2, axis=1))),
    )


def bootstrap(y, probabilities, reps, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in np.unique(y)]
    scores = {s: [] for s in probabilities}
    for _ in range(reps):
        # Same resampled IDs for all methods; fixed observed class counts.
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for stage, p in probabilities.items():
            pred = p[idx].argmax(1)
            scores[stage].append(
                [
                    accuracy_score(y[idx], pred),
                    f1_score(
                        y[idx],
                        pred,
                        labels=np.arange(10),
                        average="macro",
                        zero_division=0,
                    ),
                ]
            )
    rows = []
    for stage, arr in scores.items():
        arr = np.array(arr)
        for j, m in enumerate(["accuracy", "macro_f1"]):
            lo, hi = np.quantile(arr[:, j], [0.025, 0.975])
            rows.append(
                dict(
                    comparison=stage,
                    metric=m,
                    estimate=metrics(y, probabilities[stage])[m],
                    ci_low=lo,
                    ci_high=hi,
                    kind="single",
                )
            )
    for first, second in combinations(scores, 2):
        diff = np.array(scores[first]) - np.array(scores[second])
        for j, m in enumerate(["accuracy", "macro_f1"]):
            lo, hi = np.quantile(diff[:, j], [0.025, 0.975])
            rows.append(
                dict(
                    comparison=f"{first} - {second}",
                    metric=m,
                    estimate=metrics(y, probabilities[first])[m]
                    - metrics(y, probabilities[second])[m],
                    ci_low=lo,
                    ci_high=hi,
                    kind="paired_difference",
                )
            )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, default=ROOT / "results")
    p.add_argument("--n_bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.n_bootstrap < 2:
        raise ValueError("Need at least 2 bootstrap replicates")
    frames = {
        s: pd.read_csv(a.output_dir / s / "predictions.csv")
        .sort_values("id")
        .reset_index(drop=True)
        for s in STAGES
    }
    ref = frames["frozen"]
    y = ref.label.to_numpy()
    probs = {}
    rows = []
    strata = []
    configs = []
    for s, frame in frames.items():
        if not frame[["id", "label"]].equals(ref[["id", "label"]]):
            raise ValueError("Paired comparison requires identical heldout IDs/labels")
        if not (frame.split == "test").all() or frame.id.duplicated().any():
            raise ValueError("Invalid test membership/IDs")
        pr = frame[[f"p{k}" for k in range(10)]].to_numpy()
        probs[s] = pr
        if (
            not np.allclose(pr.sum(1), 1, atol=1e-4)
            or (pr < 0).any()
            or not np.isfinite(pr).all()
        ):
            raise ValueError("Invalid predictions")
        meta = json.loads((a.output_dir / s / "run.json").read_text())
        configs.append(meta)
        row = dict(
            strategy=s,
            **metrics(y, pr),
            trainable_parameters=meta["trainable_parameters"],
            total_seconds=meta["total_seconds"],
            train_seconds=meta.get("train_seconds", meta.get("fit_predict_seconds")),
            gpu_peak_mb=meta["gpu_peak_mb"],
            mps_allocated_mb=meta["mps_allocated_mb"],
            rss_mb=meta["rss_mb"],
            device=meta["device"],
            status=meta["status"],
            test_n=len(y),
            limited_steps=meta.get("limited_steps", False),
            max_length=meta["config"]["max_length"],
            adaptation=meta.get("adaptation", s),
        )
        rows.append(row)
        _, cal = calibration(y, pr)
        cal.to_csv(a.output_dir / s / "calibration.csv", index=False)
        pd.DataFrame(
            confusion_matrix(y, pr.argmax(1), labels=np.arange(10)),
            index=CLASSES,
            columns=CLASSES,
        ).to_csv(a.output_dir / s / "confusion_matrix.csv")
        groups = {
            "length_0_128": frame.length <= 128,
            "length_129_512": frame.length.between(129, 512),
            "length_513_plus": frame.length > 512,
        }
        groups.update({f"membrane_{v}": frame.membrane == v for v in ["M", "S", "U"]})
        groups.update({f"class_{CLASSES[k]}": frame.label == k for k in range(10)})
        for group, mask in groups.items():
            if mask.any():
                strata.append(
                    dict(
                        strategy=s,
                        group=group,
                        n=int(mask.sum()),
                        **metrics(y[mask], pr[mask]),
                    )
                )
        robust = (
            pd.read_csv(a.output_dir / s / "robust_predictions.csv")
            .sort_values("id")
            .reset_index(drop=True)
        )
        if not robust[["id", "label"]].equals(ref[["id", "label"]]):
            raise ValueError("Perturbation IDs differ")
        rp = robust[[f"p{k}" for k in range(10)]].to_numpy()
        strata.append(
            dict(
                strategy=s, group="terminal_X_mask_10_each", n=len(y), **metrics(y, rp)
            )
        )
    for key in ["dataset_sha256"]:
        if len({c.get(key) for c in configs}) != 1:
            raise ValueError(f"Incompatible runs: {key}")
    for key in ["model", "revision", "seed"]:
        if len({c["config"][key] for c in configs}) != 1:
            raise ValueError(f"Incompatible runs: {key}")
    pd.DataFrame(rows).to_csv(a.output_dir / "metrics_table.csv", index=False)
    pd.DataFrame(strata).to_csv(a.output_dir / "robustness_by_group.csv", index=False)
    ci = bootstrap(y, probs, a.n_bootstrap, a.seed)
    ci.to_csv(a.output_dir / "bootstrap_ci.csv", index=False)
    write_json(
        a.output_dir / "evaluation_summary.json",
        dict(
            n_bootstrap=a.n_bootstrap,
            ci="95% percentile, paired stratified sequence bootstrap",
            seed=a.seed,
            status=rows[0]["status"],
            limitations=[
                "Bootstrap is conditional on the observed test set/class counts and one training seed; not independent family/cluster resampling.",
                "No supplied homology cluster IDs: sequence-level CI may be optimistic.",
                "Smoke/one-batch full-FT results are pipeline checks, not strategy rankings.",
                "No test-based hyperparameter selection or temperature fitting.",
            ],
            same_preprocessing=all(
                c["config"]["max_length"] == configs[0]["config"]["max_length"]
                for c in configs
            ),
            robustness="Replace up to 10 residues at EACH terminus with X after truncation; synthetic diagnostic, not a validated biological perturbation.",
        ),
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
