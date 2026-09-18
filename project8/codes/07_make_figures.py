#!/usr/bin/env python
"""One plotting entry point for smoke and full runs; PDF/SVG + 300dpi PNG."""

from utils import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from sklearn.decomposition import PCA

NAMES = {
    "frozen": "Frozen + kNN",
    "linear": "Linear probe",
    "lora": "LoRA",
    "full": "Full FT",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, default=ROOT / "results")
    p.add_argument("--embedding_dir", type=Path, default=ROOT / "data/embeddings")
    a = p.parse_args()
    out = a.output_dir / "images"
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", palette="deep", font="DejaVu Sans", font_scale=0.95)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    df = pd.read_csv(a.output_dir / "metrics_table.csv")
    smoke = (df.status != "FULL_RUN").any() or df.limited_steps.any()
    colors = sns.color_palette("deep", 4)

    def save(fig, name):
        if smoke:
            fig.text(
                0.5,
                0.005,
                "SMOKE TEST / PLACEHOLDER - not a performance ranking",
                ha="center",
                fontsize=8,
                color="#8C4D4D",
            )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    labels = [
        NAMES[s] + (" (partial)" if ad == "partial_ft" else "")
        for s, ad in zip(df.strategy, df.adaptation)
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = df.trainable_parameters.to_numpy()
    ax.bar(labels, np.maximum(vals, 1), color=colors, width=0.65)
    ax.set(
        yscale="log",
        ylabel="Gradient-trained parameters (log scale)",
        title="Parameter efficiency",
    )
    for i, v in enumerate(vals):
        ax.text(
            i,
            max(v, 1) * 1.25,
            f"{v:,}" if v else "0 (plotted at 1)",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0.7, max(vals) * 8)
    save(fig, "fig1_parameter_efficiency")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, xcol, xlabel in zip(
        axs,
        ["trainable_parameters", "total_seconds"],
        ["Trainable parameters (0 shown at 1)", "End-to-end wall time (s)"],
    ):
        xs = np.maximum(df[xcol].to_numpy(), 1)
        ys = df.macro_f1.to_numpy()
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.scatter(x, y, color=colors[i], s=70, zorder=3)
            ax.annotate(
                labels[i],
                (x, y),
                xytext=(
                    [(10, -18), (10, 10), (-35, -16), (-35, 8)][i]
                    if xcol == "total_seconds"
                    else (5, 5)
                ),
                textcoords="offset points",
                fontsize=8,
            )
        # Non-dominated frontier, never an arbitrary line through all methods.
        frontier = []
        best = -1
        for i in np.argsort(xs):
            if ys[i] > best:
                frontier.append(i)
                best = ys[i]
        ax.plot(xs[frontier], ys[frontier], color="#666666", ls="--", lw=1)
        ax.set(
            xscale="log",
            xlabel=xlabel,
            ylabel="Macro F1",
            ylim=(0, min(1, max(ys) + 0.15)),
        )
    save(fig, "fig2_performance_cost")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", c=".6", label="Perfect calibration")
    for i, s in enumerate(df.strategy):
        cal = pd.read_csv(a.output_dir / s / "calibration.csv").dropna()
        ece = df.loc[df.strategy == s, "ece"].iloc[0]
        ax.plot(
            cal.confidence,
            cal.accuracy,
            "o-",
            color=colors[i],
            label=f"{labels[i]} (ECE {ece:.3f})",
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean top-label confidence",
        ylabel="Observed accuracy",
        title="Reliability diagram (10 equal-width bins)",
    )
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig3_calibration")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    short = [
        "Nuc",
        "Cyto",
        "Extra",
        "Mito",
        "Mem",
        "ER",
        "Chl",
        "Golgi",
        "Lys/Vac",
        "Perox",
    ]
    for ax, s in zip(axs, ["frozen", "full"]):
        cm = pd.read_csv(
            a.output_dir / s / "confusion_matrix.csv", index_col=0
        ).to_numpy()
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        sns.heatmap(
            norm,
            annot=cm,
            fmt="d",
            vmin=0,
            vmax=1,
            cmap="Blues",
            xticklabels=short,
            yticklabels=short,
            ax=ax,
            cbar=s == "full",
            cbar_kws={"label": "Row fraction"},
            annot_kws={"size": 11},
        )
        ax.set(
            title=NAMES[s] + " (counts; row-normalized color)",
            xlabel="Predicted class",
            ylabel="True class",
        )
        ax.title.set_fontsize(15)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
        ax.tick_params(axis="y", rotation=0, labelsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=11)
    save(fig, "fig4_confusion_matrices")
    emb = np.load(a.embedding_dir / "embeddings.npz")
    pca = PCA(n_components=2, svd_solver="full", random_state=42).fit(
        emb["train"].astype(np.float64)
    )
    z = np.einsum("ij,kj->ik", emb["test"] - pca.mean_, pca.components_)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite PCA projection")
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, c in enumerate(sns.color_palette("deep", 10)):
        ix = emb["y_test"] == k
        ax.scatter(z[ix, 0], z[ix, 1], s=32, alpha=0.8, color=c, label=CLASSES[k])
    ax.set(
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        title="Frozen embeddings: heldout proteins; PCA fit on train only",
    )
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    save(fig, "fig5_embedding_pca")
    pd.DataFrame(
        {"id": emb["test_ids"], "label": emb["y_test"], "PC1": z[:, 0], "PC2": z[:, 1]}
    ).to_csv(a.output_dir / "embedding_pca.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlim=(0, 12), ylim=(0, 6))
    ax.axis("off")
    boxes = {
        "data": (0.1, 2.5, 1.6, 0.8, "DeepLoc FASTA\nOfficial split"),
        "token": (2.1, 2.5, 1.6, 0.8, "QC + termini\nTokenization"),
        "emb": (4.1, 4.3, 1.7, 0.8, "Frozen ESM-2\nMean pooling"),
        "knn": (6.3, 4.9, 1.6, 0.65, "Frozen + kNN"),
        "linear": (6.3, 3.9, 1.6, 0.65, "Linear probe"),
        "lora": (6.3, 2.4, 1.6, 0.65, "ESM-2 + LoRA"),
        "full": (6.3, 0.9, 1.6, 0.65, "Full fine-tuning"),
        "eval": (8.5, 2.5, 1.5, 0.8, "Heldout metrics\nCompute cost"),
        "stats": (10.4, 2.5, 1.5, 0.8, "Class analysis\nMasking controls"),
    }
    for i, (key, (x, y, w, h, text)) in enumerate(boxes.items()):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.08,rounding_size=.12",
                facecolor=colors[
                    0
                    if key in ["data", "token"]
                    else 1
                    if key in ["emb", "knn", "linear"]
                    else 2
                    if key in ["lora", "full"]
                    else 3
                ],
                alpha=0.18,
                edgecolor=".45",
                lw=0.8,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    for u, v in [
        ("data", "token"),
        ("token", "emb"),
        ("emb", "knn"),
        ("emb", "linear"),
        ("token", "lora"),
        ("token", "full"),
        ("knn", "eval"),
        ("linear", "eval"),
        ("lora", "eval"),
        ("full", "eval"),
        ("eval", "stats"),
    ]:
        x, y, w, h, _ = boxes[u]
        xx, yy, ww, hh, _ = boxes[v]
        ax.add_patch(
            FancyArrowPatch(
                (x + w + 0.08, y + h / 2),
                (xx - 0.08, yy + hh / 2),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1,
                color=".45",
                connectionstyle="arc3,rad=0",
            )
        )
    ax.text(
        6, 5.85, "ESM-2 adaptation and evaluation workflow", ha="center", fontsize=15
    )
    save(fig, "fig6_workflow")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for s, c in zip(["linear", "lora", "full"], colors[1:]):
        h = pd.read_csv(a.output_dir / s / "history.csv")
        axs[0].plot(h.epoch, h.loss, "o-", label=NAMES[s], color=c)
        axs[1].plot(h.epoch, h.accuracy, "o-", label=NAMES[s], color=c)
    axs[0].set(
        xlabel="Epoch (full FT smoke may be partial)" if smoke else "Epoch",
        ylabel="Training cross-entropy",
    )
    axs[1].set(xlabel="Epoch", ylabel="Training accuracy")
    axs[1].legend()
    save(fig, "fig7_training_curves")
    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
