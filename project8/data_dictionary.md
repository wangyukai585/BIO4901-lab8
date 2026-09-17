# Data dictionary

Raw FASTA: official DeepLoc-1.0 download; **never commit**. Raw count 14,004; exclude 146 `Cytoplasm-Nucleus` entries (all in train), retaining 13,858 single-label entries. The remaining official split is 11,085 train / 2,773 test. `Plastid` -> `Chloroplast` is the course naming convention; original annotation is broader. Dot separators become spaces. Unknown labels fail closed.

## Class order (immutable probability columns)

| label | location |
|--:|:--|
|0|Nucleus|
|1|Cytoplasm|
|2|Extracellular|
|3|Mitochondrion|
|4|Cell membrane|
|5|Endoplasmic reticulum|
|6|Chloroplast (source: Plastid)|
|7|Golgi apparatus|
|8|Lysosome/Vacuole|
|9|Peroxisome|

## Fields

- `id`: UniProt accession from header; unique.
- `sequence`: uppercase amino acids; only in ignored raw/processed files.
- `split`: official `test` token -> test, otherwise train; never reassigned.
- `membrane`: S soluble, M membrane, U unknown; not a localization class.
- `length`: original residue count, before truncation/perturbation.
- `sequence_hash`: SHA-256 complete sequence; duplicate audit only.
- `p0` ... `p9`: probabilities in class order above; sum to one.
- `prediction`: argmax probability class index.
- `trainable_parameters`: scalar parameters with requires_grad; kNN zero, but stores reference embeddings.
- `gpu_peak_mb`: CUDA peak allocated bytes / 2^20; null on Mac, never zero-filled as a measurement.
- `mps_allocated_mb`: MPS current allocation at recording time, **not peak GPU memory**.
- `rss_mb`: host process resident memory, not device memory.
- `total_seconds`: end-to-end wall clock including model loading, training, evaluation; Linear includes frozen extraction cost, excludes kNN fitting. Network cache affects costs.
- `train_seconds`: training loop including epoch checkpoint I/O; Frozen classifier fit+prediction seconds.
- `status`: SMOKE_PLACEHOLDER or FULL_RUN; inspect `limited_steps` independently.
- `adaptation`: full becomes partial_ft when early blocks are frozen.
- `ECE`: sum over confidence bins of n_bin/n × abs(accuracy_bin-confidence_bin).
- `NLL`: scikit-learn clipped multiclass log loss; kNN zero probabilities can make NLL high.
- `Brier`: mean sum of squared probability errors across all 10 classes (not divided by 10).
- `macro_f1`: arithmetic mean over fixed 10 classes; absent-class F1 = 0.
- bootstrap `comparison`: method or first minus second; same resampled IDs across methods.

`results/qc/summary.json` supplies full counts, imbalance ratio, truncation fractions, source SHA-256 and exclusion policy. `run.json` supplies exact configuration and hardware. `robustness_by_group.csv` includes class, length and S/M/U groups and synthetic terminal masking. `checkpoint_check.json` records fixed-batch loss and parameter restore checks. Sequence-level bootstrap does not establish independence of homologous proteins.
