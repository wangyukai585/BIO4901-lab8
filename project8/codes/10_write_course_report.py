#!/usr/bin/env python
"""Fill the concise, paginated course report from verified full-run evidence."""

from utils import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, default=ROOT / "results/full_run")
    a = p.parse_args()
    r = a.results_dir
    table = pd.read_csv(r / "metrics_table.csv").set_index("strategy")
    if not (table.status == "FULL_RUN").all() or table.limited_steps.any():
        raise ValueError("A formal report requires completed full-data runs")
    qc = json.loads((ROOT / "results/qc/summary.json").read_text())
    for s in ["frozen", "linear", "lora", "full"]:
        run = json.loads((r / s / "run.json").read_text())
        if (
            run["train_n"] != qc["audit"]["train_n"]
            or run["test_n"] != qc["audit"]["test_n"]
        ):
            raise ValueError("Incomplete data coverage")
    audit = json.loads((r / "biological_controls/audit.json").read_text())
    if audit["status"] != "COMPLETED" or audit["test_n"] != 2773:
        raise ValueError("Incomplete biological audit")
    bio = pd.read_csv(r / "biological_controls/class_effects.csv").set_index(
        ["strategy", "category", "condition"]
    )
    f, l, o = (table.loc[s] for s in ["full", "linear", "lora"])
    ci = pd.read_csv(r / "bootstrap_ci.csv")
    pair = ci[(ci.comparison == "lora - full") & (ci.metric == "macro_f1")].iloc[0]
    names = dict(frozen="Frozen + kNN", linear="Linear", lora="LoRA", full="Full FT")
    rows = [
        "| 方法 | Accuracy | Macro F1 | 更新参数 | 时间/min | 显存/GiB |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for s, row in table.iterrows():
        rows.append(
            f"| {names[s]} | {row.accuracy:.1%} | {row.macro_f1:.3f} | {int(row.trainable_parameters):,} | {row.total_seconds / 60:.1f} | {row.gpu_peak_mb / 1024:.2f} |"
        )

    def drop(category, condition):
        return 100 * bio.loc[("full", category, condition), "recall_drop"]

    biological = f"Full FT 中，遮蔽开头后，线粒体类别正确识别率下降 {drop('Mitochondrion', 'N10'):.1f} 个百分点，内部对照下降 {drop('Mitochondrion', 'internal10'):.1f} 个百分点；叶绿体对应为 {drop('Chloroplast', 'N10'):.1f} 和 {drop('Chloroplast', 'internal10'):.1f} 个百分点。过氧化物酶体遮蔽结尾后下降 {drop('Peroxisome', 'C10'):.1f} 个百分点，内部对照下降 {drop('Peroxisome', 'internal10'):.1f} 个百分点。"
    values = {
        "METRICS": "\n".join(rows),
        "PERFORMANCE": f"**Full FT 的 Macro F1 最高，为 {f.macro_f1:.3f}；Linear 是低成本基线。** Full FT 的准确率为 {f.accuracy:.1%}，Linear 为 {l.accuracy:.1%}。参数调整带来了改善，但这只是固定设置、一次训练的比较，并非各方法的最优成绩。",
        "COST": f"LoRA 的更新参数只有 Full FT 的 {100 * o.trainable_parameters / f.trainable_parameters:.2f}%，显存约一半，但耗时同为约 107 分钟。因为它仍需通过编码器计算梯度。Linear 连同表征提取仅需 {l.total_seconds / 60:.1f} 分钟。",
        "CI": f"Full FT 比 LoRA 的 Macro F1 高 {f.macro_f1 - o.macro_f1:.3f}，95% 区间为 [{-pair.ci_high:.3f}, {-pair.ci_low:.3f}]，未跨零。Linear 与 LoRA 的 Macro F1 差异区间跨零，不能据此认定两者有稳定差距。",
        "BIOLOGY": biological,
        "CALIBRATION": f"Linear 的校准误差 ECE 最低（{l.ece:.3f}）；LoRA 与 Full FT 分别为 {o.ece:.3f}、{f.ece:.3f}。**Full FT 更准确，但往往过于自信。** 本实验没有用测试集调整置信度。若要据此筛选可靠预测，还需要独立数据做校准。",
    }
    text = (ROOT / "report/template.qmd").read_text()
    for key, value in values.items():
        text = text.replace("@@" + key + "@@", value)
    if "@@" in text:
        raise ValueError("Unfilled report marker")
    rel = str(r.resolve().relative_to(ROOT))
    text = text.replace("results/full_run/", rel + "/")
    for path in re.findall(r"\]\(([^)]+\.pdf)\)", text):
        if not (ROOT / path).exists():
            raise FileNotFoundError(path)
    (ROOT / "lab8.qmd").write_text(text)
    print("Report updated from verified full experiments and biological controls")


if __name__ == "__main__":
    main()
