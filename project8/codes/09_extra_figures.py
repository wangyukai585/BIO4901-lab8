#!/usr/bin/env python
"""Readable class-level, robustness and uncertainty figures for the course report."""

from utils import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

NAMES = {
    "frozen": "Frozen + kNN",
    "linear": "Linear probe",
    "lora": "LoRA",
    "full": "Full FT",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, default=ROOT / "results")
    a = p.parse_args()
    out = a.output_dir / "images"
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(a.output_dir / "metrics_table.csv")
    groups = pd.read_csv(a.output_dir / "robustness_by_group.csv")
    smoke = (table.status != "FULL_RUN").any() or table.limited_steps.any()
    sns.set_theme(style="ticks", palette="deep", font="DejaVu Sans")
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = sns.color_palette("deep", 4)
    stages = list(NAMES)

    def save(fig, name):
        if smoke:
            fig.text(
                0.5,
                0.005,
                "SMOKE / PLACEHOLDER - not a performance ranking",
                ha="center",
                fontsize=8,
                color="#8c4d4d",
            )
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Per-class F1 from the original confusion matrix, not fixed-ten macro-F1 of a one-class subset.
    scores = []
    counts = None
    for s in stages:
        cm = pd.read_csv(
            a.output_dir / s / "confusion_matrix.csv", index_col=0
        ).to_numpy()
        denom = cm.sum(0) + cm.sum(1)
        scores.append(
            np.divide(
                2 * cm.diagonal(), denom, out=np.zeros(10, dtype=float), where=denom > 0
            )
        )
        counts = cm.sum(1)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(
        np.array(scores).T,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="YlGnBu",
        xticklabels=list(NAMES.values()),
        yticklabels=[f"{c} (n={n})" for c, n in zip(CLASSES, counts)],
        cbar_kws={"label": "Class F1"},
        ax=ax,
    )
    ax.set_title("Which classes benefit from adaptation?")
    ax.tick_params(axis="y", rotation=0)
    save(fig, "fig8_class_f1")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    length_keys = ["length_0_128", "length_129_512", "length_513_plus"]
    for s, c in zip(stages, colors):
        g = groups[groups.strategy == s].set_index("group")
        axs[0].plot(
            range(3),
            [g.loc[k, "accuracy"] if k in g.index else np.nan for k in length_keys],
            "o-",
            color=c,
            label=NAMES[s],
        )
        clean = float(table.loc[table.strategy == s, "accuracy"].iloc[0])
        changed = float(g.loc["terminal_X_mask_10_each", "accuracy"])
        axs[1].plot([0, 1], [clean, changed], "o-", color=c, label=NAMES[s])
    axs[0].set(
        xticks=range(3),
        xticklabels=["1-128", "129-512", ">512"],
        xlabel="Original sequence length (residues)",
        ylabel="Accuracy",
        ylim=(0, 1),
        title="Generalization across sequence lengths",
    )
    axs[1].set(
        xticks=[0, 1],
        xticklabels=["Original", "Termini masked"],
        ylabel="Accuracy",
        ylim=(0, 1),
        title="Sensitivity to terminal information",
    )
    axs[1].legend(fontsize=8)
    sns.despine()
    save(fig, "fig9_robustness")
    ci = pd.read_csv(a.output_dir / "bootstrap_ci.csv")
    selected = ci[
        (ci.kind == "paired_difference") & (ci.metric == "macro_f1")
    ].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(selected.itertuples()):
        ax.plot([r.ci_low, r.ci_high], [i, i], color=colors[0], lw=2)
        ax.scatter(r.estimate, i, color=colors[0], s=35, zorder=3)
    ax.axvline(0, ls="--", color=".5", lw=1)
    ax.set(
        yticks=range(len(selected)),
        yticklabels=selected.comparison,
        xlabel="Macro F1 difference (first - second)",
        title="Paired bootstrap: 95% percentile intervals",
    )
    ax.invert_yaxis()
    sns.despine()
    save(fig, "fig10_paired_intervals")
    print(f"Additional figures written to {out}")


if __name__ == "__main__":
    main()
