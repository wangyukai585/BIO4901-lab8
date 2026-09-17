"""Shared linear/LoRA/full training, resumable checkpoints, CUDA preflight."""

import copy, math, time
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
from utils import *


class ProteinClassifier(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(backbone.config.hidden_size, 10)

    def forward(self, batch):
        return self.head(pooled(self.backbone, batch))


def build_model(a, stage):
    tok, base = load_backbone(a)
    if stage == "lora":
        base = get_peft_model(
            base,
            LoraConfig(
                r=a.lora_rank,
                lora_alpha=2 * a.lora_rank,
                lora_dropout=0.05,
                target_modules=["query", "value"],
                bias="none",
            ),
        )
    elif stage == "full":
        for block in base.encoder.layer[: a.freeze_layers]:
            for p in block.parameters():
                p.requires_grad = False
        if a.freeze_layers:
            for p in base.embeddings.parameters():
                p.requires_grad = False
    if a.gradient_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return tok, ProteinClassifier(base)


def optimizer(model, a, d):
    ps = [p for p in model.parameters() if p.requires_grad]
    if d.type == "cuda" and a.optimizer == "adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(ps, lr=a.lr, weight_decay=0.01)
    return torch.optim.AdamW(ps, lr=a.lr, weight_decay=0.01)


def is_oom(e):
    return "out of memory" in str(e).lower()


def next_fallback(a, n_layers=30):
    """Ordered descent, recorded explicitly; partial FT is not labeled full FT."""
    if a.batch_size > 1:
        a.batch_size = max(1, a.batch_size // 2)
        return "reduce_batch"
    if a.max_length > a.min_length:
        a.max_length = max(a.min_length, a.max_length // 2)
        return "reduce_length"
    if a.freeze_layers < n_layers - 1:
        a.freeze_layers = min(n_layers - 1, a.freeze_layers + 5)
        return "freeze_early_layers"
    return None


def probe_memory(a, d, log):
    """Probe real worst configured shape incl. optimizer state, discard all updates.
    This is a measured preflight, not a guarantee; no linear extrapolation from tiny inputs.
    """
    attempts = []
    while True:
        model = opt = b = loss = None
        try:
            seed_all(a.seed)
            tok, model = build_model(a, "full")
            model.to(d)
            model.train()
            opt = optimizer(model, a, d)
            if d.type == "cuda":
                torch.cuda.reset_peak_memory_stats(d)
            started = time.monotonic()
            for length, batch in [
                (min(16, a.max_length), 1),
                (a.max_length, a.batch_size),
            ]:
                b = tokenize(tok, ["A" * length] * batch, a.max_length, d)
                with amp(d, precision(d, a.precision)):
                    loss = torch.nn.functional.cross_entropy(
                        model(b), torch.zeros(batch, dtype=torch.long, device=d)
                    )
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                sync(d)
            attempts.append(
                dict(
                    batch_size=a.batch_size,
                    max_length=a.max_length,
                    freeze_layers=a.freeze_layers,
                    success=True,
                    seconds=time.monotonic() - started,
                    **memory(d),
                )
            )
            log.info("memory_probe=%s", attempts[-1])
            break
        except RuntimeError as e:
            if not is_oom(e):
                raise
            attempts.append(
                dict(
                    batch_size=a.batch_size,
                    max_length=a.max_length,
                    freeze_layers=a.freeze_layers,
                    success=False,
                    error=str(e),
                )
            )
            action = next_fallback(a)
            log.warning("OOM fallback: %s", action)
            if action is None:
                raise RuntimeError(
                    "All memory fallback configurations exhausted"
                ) from None
        finally:
            del model, opt, b, loss
            clear_device(d)
    return attempts


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    )


def set_rng(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    if s["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(s["mps"])


def checkpoint(path, model, opt, epoch, step, a, history):
    # Only trainable state for PEFT/linear; frozen base is pinned by revision.
    trainable = {k for k, p in model.named_parameters() if p.requires_grad}
    state = {
        k: v.detach().cpu() for k, v in model.state_dict().items() if k in trainable
    }
    tmp = path.with_suffix(".tmp.pt")
    torch.save(
        dict(
            model=state,
            optimizer=opt.state_dict(),
            epoch=epoch,
            step=step,
            config=vars(a),
            dataset_sha256=data_digest(a.data) if hasattr(a, "data") else None,
            history=history,
            rng=rng_state(),
        ),
        tmp,
    )
    tmp.replace(path)
    write_json(
        path.with_suffix(".json"),
        dict(
            epoch=epoch,
            step=step,
            config=vars(a),
            state_keys=len(state),
            checkpoint_policy="epoch boundary; optimizer + RNG + trainable tensors",
        ),
    )


def restore(path, model, opt=None):
    state = torch.load(
        path, map_location="cpu", weights_only=False
    )  # Only trusted local checkpoints.
    mismatch = model.load_state_dict(state["model"], strict=False)
    if mismatch.unexpected_keys:
        raise ValueError(mismatch.unexpected_keys)
    expected = {k for k, p in model.named_parameters() if p.requires_grad}
    if not expected.issubset(state["model"]):
        raise ValueError("Missing trainable checkpoint tensors")
    if opt is not None:
        opt.load_state_dict(state["optimizer"])
        set_rng(state["rng"])
    return state


def main(stage):
    p = parser(stage)
    p.add_argument("--embedding_dir", type=Path, default=ROOT / "data/embeddings")
    p.add_argument(
        "--lr",
        type=float,
        default=0.001 if stage == "linear" else 1e-4 if stage == "lora" else 1e-5,
    )
    p.add_argument("--effective_batch_size", type=int, default=16)
    p.add_argument(
        "--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    p.add_argument(
        "--optimizer",
        choices=["adamw", "adamw_8bit"],
        default="adamw" if stage == "linear" else "adamw_8bit",
    )
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--freeze_layers", type=int, default=0)
    p.add_argument("--min_length", type=int, default=128)
    p.add_argument("--probe_only", action="store_true")
    p.add_argument("--skip_probe", action="store_true")
    p.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optimizer steps; 1 for local full-FT smoke",
    )
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    if (
        min(a.epochs, a.batch_size, a.effective_batch_size, a.log_steps) < 1
        or not 2 <= a.max_length <= 1022
    ):
        raise ValueError("Invalid positive count/length")
    if not 0 <= a.freeze_layers <= 29 or not 2 <= a.min_length <= a.max_length:
        raise ValueError("Invalid freeze_layers/min_length")
    if a.max_steps is not None and a.max_steps < 1:
        raise ValueError("max_steps must be positive")
    seed_all(a.seed)
    d = get_device(a.device)
    log = logger(a.output_dir)
    if d.type != "cuda" and a.optimizer == "adamw_8bit":
        log.info(
            "8-bit optimizer is CUDA-only in pinned version; using fp32 AdamW locally"
        )
    log.info(
        "precision=%s (bf16 after capability check; otherwise fp32 fallback)",
        precision(d, a.precision),
    )
    start = time.monotonic()
    attempts = []
    if stage == "full" and not a.skip_probe and (d.type == "cuda" or a.probe_only):
        with Monitor(log, d, 1) as mon:
            attempts = probe_memory(a, d, log)
        write_json(a.output_dir / "memory_probe.json", attempts)
    if a.probe_only:
        return
    tr, te = load_data(a)
    if a.resume:
        resume_state = torch.load(a.resume, map_location="cpu", weights_only=False)
        cfg = resume_state["config"]
        if resume_state.get("dataset_sha256") != data_digest(a.data):
            raise ValueError("Resume dataset content mismatch")
        del resume_state
        for key in [
            "model",
            "revision",
            "max_length",
            "batch_size",
            "effective_batch_size",
            "seed",
            "subset_size",
            "eval_size",
            "freeze_layers",
            "lora_rank",
            "lr",
            "optimizer",
            "gradient_checkpointing",
            "precision",
        ]:
            if cfg[key] != getattr(a, key):
                raise ValueError(f"Resume configuration mismatch: {key}")
    train_x = test_x = robust_x = None
    extraction = 0.0
    if stage == "linear":
        emb = np.load(a.embedding_dir / "embeddings.npz")
        meta = json.loads((a.embedding_dir / "metadata.json").read_text())
        if meta.get("dataset_sha256") != data_digest(a.data):
            raise ValueError("Embedding dataset content mismatch")
        for k in ["model", "revision", "max_length", "seed"]:
            if str(meta["config"][k]) != str(getattr(a, k)):
                raise ValueError(f"Embedding config mismatch: {k}")
        if not np.array_equal(emb["train_ids"], tr.id) or not np.array_equal(
            emb["test_ids"], te.id
        ):
            raise ValueError("Embedding sample IDs do not match")
        train_x = torch.tensor(emb["train"])
        test_x = torch.tensor(emb["test"])
        robust_x = torch.tensor(emb["robust"])
        model = torch.nn.Linear(train_x.shape[1], 10)
        tok = None
        extraction = meta["extraction_seconds"]
        meta_config_peak = meta["gpu_peak_mb"] or 0
    else:
        with Monitor(log, d, 1) as mon:
            tok, model = build_model(a, stage)
    model.to(d)
    opt = optimizer(model, a, d)
    dtype = precision(d, a.precision)
    if d.type == "cuda":
        torch.cuda.reset_peak_memory_stats(d)
    history = []
    start_epoch = 0
    step = 0
    if a.resume:
        ck = restore(a.resume, model, opt)
        start_epoch = ck["epoch"]
        step = ck["step"]
        history = ck["history"]
    # Round up preserves target effective batch size when batch is reduced.
    accum = max(1, math.ceil(a.effective_batch_size / a.batch_size))
    batches = math.ceil(len(tr) / a.batch_size)
    planned = math.ceil(batches / accum) * a.epochs
    if a.max_steps:
        planned = min(planned, a.max_steps)
    train_start = time.monotonic()
    grad_checked = False
    initial_loss = None

    def get_batch(indices, frame=tr, x=train_x, perturb=False):
        return (
            x[indices].to(d)
            if x is not None
            else tokenize(
                tok, frame.iloc[indices].sequence.tolist(), a.max_length, d, perturb
            )
        )

    fixed_idx = np.arange(min(a.batch_size, len(tr)))
    model.eval()
    with torch.no_grad(), amp(d, dtype):
        initial_loss = float(
            torch.nn.functional.cross_entropy(
                model(get_batch(fixed_idx)),
                torch.tensor(tr.label.iloc[fixed_idx].tolist(), device=d),
            )
        )
    try:
        with Monitor(log, d, planned, estimate=5.0 * accum) as mon:
            for epoch in range(start_epoch, a.epochs):
                if a.max_steps and step >= a.max_steps:
                    break
                model.train()
                perm = np.random.default_rng(a.seed + epoch).permutation(len(tr))
                loss_sum = correct = seen = 0
                for group_start in tqdm(
                    range(0, batches, accum),
                    desc=f"{stage} epoch {epoch + 1}/{a.epochs}",
                ):
                    opt.zero_grad(set_to_none=True)
                    group_end = min(batches, group_start + accum)
                    group_n = (
                        min(len(tr), group_end * a.batch_size)
                        - group_start * a.batch_size
                    )
                    for j in range(group_start, group_end):
                        idx = perm[j * a.batch_size : (j + 1) * a.batch_size]
                        y = torch.tensor(tr.label.iloc[idx].tolist(), device=d)
                        with amp(d, dtype):
                            logits = model(get_batch(idx))
                            raw_loss = torch.nn.functional.cross_entropy(logits, y)
                            loss = raw_loss * len(idx) / group_n
                        if not torch.isfinite(raw_loss):
                            raise FloatingPointError("Nonfinite loss")
                        loss.backward()
                        loss_sum += raw_loss.item() * len(idx)
                        correct += (logits.argmax(1) == y).sum().item()
                        seen += len(idx)
                    if not grad_checked:
                        grads = {
                            n: float(p.grad.float().norm())
                            for n, p in model.named_parameters()
                            if p.grad is not None
                        }
                        if (
                            not grads
                            or not all(np.isfinite(v) for v in grads.values())
                            or not any(v > 0 for v in grads.values())
                        ):
                            raise AssertionError("Gradient failure")
                        if stage == "lora" and not any(
                            "lora_B" in n and v > 0 for n, v in grads.items()
                        ):
                            raise AssertionError("LoRA gradients absent")
                        write_json(
                            a.output_dir / "gradient_check.json",
                            dict(
                                nonzero_tensor_count=sum(v > 0 for v in grads.values()),
                                tensor_count=len(grads),
                                lora_nonzero=sum(
                                    "lora_" in n and v > 0 for n, v in grads.items()
                                ),
                                all_finite=True,
                            ),
                        )
                        grad_checked = True
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0
                    )
                    opt.step()
                    step += 1
                    if step % a.log_steps == 0 or step == planned:
                        mon.update(
                            step,
                            epoch=epoch + 1,
                            loss=loss_sum / seen,
                            accuracy=correct / seen,
                        )
                    if a.max_steps and step >= a.max_steps:
                        break
                history.append(
                    dict(
                        epoch=epoch + 1,
                        step=step,
                        loss=loss_sum / seen,
                        accuracy=correct / seen,
                        examples_seen=seen,
                    )
                )
                checkpoint(
                    a.output_dir / "last.pt", model, opt, epoch + 1, step, a, history
                )
                pd.DataFrame(history).to_csv(a.output_dir / "history.csv", index=False)
                if a.max_steps and step >= a.max_steps:
                    break
    except RuntimeError as e:
        if stage == "full" and is_oom(e):
            # Restart from pinned base (never reuse partially updated OOM state).
            write_json(
                a.output_dir / "runtime_oom.json",
                dict(
                    error=str(e),
                    action="rerun preflight with next smaller configuration",
                    config=vars(a),
                ),
            )
            action = next_fallback(a)
            if action:
                log.warning(
                    "Runtime OOM: restart entire stage with %s; discarded partial run",
                    action,
                )
                cmd = [
                    sys.executable,
                    str(Path(__file__).with_name("05_train_full_ft.py")),
                ]
                for k, v in vars(a).items():
                    if k == "resume" or v is None:
                        continue
                    if isinstance(v, bool):
                        if k == "gradient_checkpointing":
                            cmd.append(
                                "--gradient_checkpointing"
                                if v
                                else "--no-gradient_checkpointing"
                            )
                        elif v:
                            cmd.append("--" + k)
                    else:
                        cmd += ["--" + k, str(v)]
                # Replace process to release ALL graph/optimizer/device allocations before retry.
                for handler in log.handlers:
                    handler.flush()
                os.execv(sys.executable, cmd)
        raise
    sync(d)
    train_seconds = time.monotonic() - train_start
    model.eval()
    with torch.no_grad(), amp(d, dtype):
        fixed_input = get_batch(fixed_idx)
        before = model(fixed_input).float().cpu()
        final_loss = float(
            torch.nn.functional.cross_entropy(
                before, torch.tensor(tr.label.iloc[fixed_idx].tolist())
            )
        )
    # Perturb and restore head to validate load, not merely re-read identical state.
    with torch.no_grad():
        next(p for p in model.parameters() if p.requires_grad).add_(0.01)
    restore(a.output_dir / "last.pt", model)
    model.eval()
    with torch.no_grad(), amp(d, dtype):
        after = model(fixed_input).float().cpu()
    torch.testing.assert_close(before, after, rtol=1e-4, atol=1e-5)
    write_json(
        a.output_dir / "checkpoint_check.json",
        dict(
            roundtrip=True,
            initial_fixed_batch_loss=initial_loss,
            final_fixed_batch_loss=final_loss,
            loss_decreased=final_loss < initial_loss,
        ),
    )
    with Monitor(log, d, 2 * math.ceil(len(te) / a.batch_size)) as mon:
        for perturb, x, name in [
            (False, test_x, "predictions.csv"),
            (True, robust_x, "robust_predictions.csv"),
        ]:
            probs = []
            with torch.inference_mode():
                for i in tqdm(
                    range(0, len(te), a.batch_size),
                    desc="Evaluate heldout" + (" perturbed" if perturb else ""),
                ):
                    b = get_batch(
                        np.arange(i, min(i + a.batch_size, len(te))), te, x, perturb
                    )
                    with amp(d, dtype):
                        pr = model(b).float().softmax(-1).cpu().numpy()
                    probs.append(pr)
                    mon.done += 1
            save_predictions(a.output_dir / name, te, np.concatenate(probs))
    meta = run_metadata(
        a,
        d,
        stage=stage,
        adaptation="partial_ft" if stage == "full" and a.freeze_layers else stage,
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        total_parameters=sum(p.numel() for p in model.parameters()),
        train_seconds=train_seconds,
        shared_extraction_seconds=extraction,
        total_seconds=time.monotonic() - start + extraction,
        optimizer_actual=type(opt).__name__,
        precision_actual=str(dtype or "fp32"),
        effective_batch_size=accum * a.batch_size,
        train_n=len(tr),
        test_n=len(te),
        final_training_loss=history[-1]["loss"],
        final_training_accuracy=history[-1]["accuracy"],
        final_test_accuracy=float(
            (
                pd.read_csv(a.output_dir / "predictions.csv").prediction.to_numpy()
                == te.label.to_numpy()
            ).mean()
        ),
        optimizer_steps=step,
        limited_steps=a.max_steps is not None,
        probe_attempts=attempts,
    )
    if stage == "linear":
        meta["head_only_gpu_peak_mb"] = meta["gpu_peak_mb"]
        meta["gpu_peak_mb"] = (
            max(meta["gpu_peak_mb"] or 0, meta_config_peak)
            if d.type == "cuda"
            else None
        )
    write_json(a.output_dir / "run.json", meta)
    log.info("Completed %s: %s", stage, meta)
