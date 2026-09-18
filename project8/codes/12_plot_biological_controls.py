#!/usr/bin/env python
"""Plot matched-length masking results; biological expectations are hypotheses."""

from utils import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, default=ROOT / "results/full_run")
    a = p.parse_args()
    df = pd.read_csv(a.results_dir / "biological_controls/class_effects.csv")
    sns.set_theme(style="ticks", palette="deep", font="DejaVu Sans", font_scale=1.05)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    stages = ["frozen", "linear", "lora", "full"]
    names = ["Frozen", "Linear", "LoRA", "Full FT"]
    conditions = ["N10", "C10", "internal10"]
    labels = ["N-terminus", "C-terminus", "Internal control"]
    colors = sns.color_palette("deep", 3)
    for ax, category in zip(axs, ["Mitochondrion", "Chloroplast", "Peroxisome"]):
        sub = df[df.category == category]
        for i, (condition, label, color) in enumerate(zip(conditions, labels, colors)):
            ss = sub[sub.condition == condition].set_index("strategy").loc[stages]
            x = np.arange(4) + (i - 1) * 0.23
            ax.bar(x, 100 * ss.recall_drop, width=0.21, color=color, label=label)
        ax.axhline(0, color=".4", lw=0.7)
        ax.set(
            title=f"{category} (n={int(sub.n.iloc[0])})",
            xticks=np.arange(4),
            xticklabels=names,
        )
        ax.tick_params(axis="x", rotation=20)
    axs[0].set_ylabel("Class recall drop (percentage points)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = a.results_dir / "images"
    out.mkdir(exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(
            out / f"fig11_biological_controls.{ext}", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
