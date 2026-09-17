#!/usr/bin/env python
"""Frozen ESM mean embeddings + 5-nearest-neighbor reference baseline."""

import math, time
from tqdm.auto import tqdm
from frozen_knn import FrozenKNN
import joblib
from utils import *


def extract(model, tok, df, a, d, mon, offset=0, perturb=False):
    chunks = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(df), a.batch_size), desc="Frozen embeddings"):
            b = tokenize(
                tok,
                df.sequence.iloc[i : i + a.batch_size].tolist(),
                a.max_length,
                d,
                perturb,
            )
            chunks.append(pooled(model, b).float().cpu().numpy())
            step = offset + i // a.batch_size + 1
            if step % a.log_steps == 0:
                mon.update(step, phase="extract")
    return np.concatenate(chunks)


def main():
    p = parser("frozen")
    p.add_argument("--embedding_dir", type=Path, default=ROOT / "data/embeddings")
    a = p.parse_args()
    seed_all(a.seed)
    d = get_device(a.device)
    log = logger(a.output_dir)
    start = time.monotonic()
    if not 2 <= a.max_length <= 1022 or a.batch_size < 1:
        raise ValueError("Invalid length/batch")
    tr, te = load_data(a)
    a.embedding_dir.mkdir(parents=True, exist_ok=True)
    total = math.ceil(len(tr) / a.batch_size) + 2 * math.ceil(len(te) / a.batch_size)
    with Monitor(log, d, total) as mon:
        tok, model = load_backbone(a)
        model.to(d)
        for param in model.parameters():
            param.requires_grad = False
        train = extract(model, tok, tr, a, d, mon)
        test = extract(model, tok, te, a, d, mon, math.ceil(len(tr) / a.batch_size))
        robust = extract(
            model,
            tok,
            te,
            a,
            d,
            mon,
            math.ceil(len(tr) / a.batch_size) + math.ceil(len(te) / a.batch_size),
            True,
        )
    sync(d)
    extraction = time.monotonic() - start
    fit_start = time.monotonic()
    clf = FrozenKNN(k=5).fit(train, tr.label)
    joblib.dump(clf, a.output_dir / "knn.joblib")
    probs = clf.predict_proba(test)
    rprobs = clf.predict_proba(robust)
    if not np.array_equal(clf.classes_, np.arange(10)):
        raise ValueError("All train classes required")
    save_predictions(a.output_dir / "predictions.csv", te, probs)
    save_predictions(a.output_dir / "robust_predictions.csv", te, rprobs)
    # Arrays are ignored by git; metadata keeps identity and preprocessing provenance.
    np.savez_compressed(
        a.embedding_dir / "embeddings.npz",
        train=train,
        test=test,
        robust=robust,
        y_train=tr.label.to_numpy(),
        y_test=te.label.to_numpy(),
        train_ids=tr.id.to_numpy(dtype=str),
        test_ids=te.id.to_numpy(dtype=str),
    )
    metadata = run_metadata(
        a,
        d,
        trainable_parameters=0,
        total_parameters=sum(p.numel() for p in model.parameters()),
        final_test_accuracy=float((probs.argmax(1) == te.label.to_numpy()).mean()),
        classifier="5-NN cosine distance; no gradient-trained parameters",
        extraction_seconds=extraction,
        fit_predict_seconds=time.monotonic() - fit_start,
        total_seconds=time.monotonic() - start,
        train_n=len(tr),
        test_n=len(te),
    )
    write_json(a.embedding_dir / "metadata.json", metadata)
    write_json(a.output_dir / "run.json", metadata)
    log.info("Completed frozen: %s", metadata)


if __name__ == "__main__":
    main()
