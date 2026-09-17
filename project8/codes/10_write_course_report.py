#!/usr/bin/env python
"""Build a concise Chinese course report from verified completed server results."""

from utils import *

NAMES = {
    "frozen": "Frozen + kNN",
    "linear": "Linear probing",
    "lora": "LoRA",
    "full": "Full fine-tuning",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=Path, default=ROOT / "results/full_run")
    a = p.parse_args()
    r = a.results_dir
    table = pd.read_csv(r / "metrics_table.csv").set_index("strategy")
    ci = pd.read_csv(r / "bootstrap_ci.csv")
    qc = json.loads((ROOT / "results/qc/summary.json").read_text())
    if not (table.status == "FULL_RUN").all() or table.limited_steps.any():
        raise ValueError("A formal report requires completed full-data runs")
    runs = {s: json.loads((r / s / "run.json").read_text()) for s in NAMES}
    hardware = json.loads((r / "server_environment.json").read_text())
    if any(
        m["train_n"] != qc["audit"]["train_n"] or m["test_n"] != qc["audit"]["test_n"]
        for m in runs.values()
    ):
        raise ValueError("Incomplete data coverage")
    robust = pd.read_csv(r / "robustness_by_group.csv").set_index(["strategy", "group"])
    f, l, a = (table.loc[x] for x in ("full", "linear", "lora"))
    pair = ci[(ci.comparison == "lora - full") & (ci.metric == "macro_f1")].iloc[0]
    cm = pd.read_csv(r / "full/confusion_matrix.csv", index_col=0)
    per_class = pd.Series(
        2 * np.diag(cm) / (cm.sum(axis=0).values + cm.sum(axis=1).values),
        index=cm.index,
    )
    best = table.macro_f1.idxmax()
    base = table.loc["frozen"]
    bestrow = table.loc[best]
    rel = r.resolve().relative_to(ROOT)

    def figure(name, caption, width=95):
        if not (r / "images" / f"{name}.pdf").exists():
            raise FileNotFoundError(name)
        return (
            f"![{caption}]({rel}/images/{name}.pdf)"
            + "{width="
            + str(width)
            + '% fig-pos="H"}'
        )

    lines = [
        r"""---
title: "BIO4901 Lab 8：用 ESM-2 预测蛋白质亚细胞定位"
author: "Yukai Wang"
lang: zh
format:
  pdf:
    pdf-engine: xelatex
    documentclass: article
    papersize: a4
    fontsize: 10pt
    geometry: margin=22mm
    number-sections: true
    colorlinks: true
    mainfont: "Times New Roman"
    CJKmainfont: "PingFang SC"
    include-in-header:
      text: |
        \usepackage[ruled,vlined]{algorithm2e}
        \usepackage{float}
        \IfFileExists{bio4901.sty}{\usepackage{bio4901}}{}
execute:
  enabled: false
---

# 我们想解决什么问题？

蛋白质需要到达正确的细胞位置才能发挥作用。这个实验尝试只根据氨基酸序列，判断蛋白质属于细胞核、线粒体等哪一类位置。我们使用预训练的 ESM-2 150M，比较四种使用方式：直接使用固定表征、只训练线性分类层、LoRA，以及全参数微调。

重点不只是看谁分数高，也看提升需要多少参数、时间和显存，以及模型在少数类别、不同长度序列上的表现。

# 数据和实验方法
""",
        f"DeepLoc-1.0 官方文件有 {qc['raw_n']:,} 条序列。我们排除 {qc['excluded_dual_localization']} 条 `Cytoplasm-Nucleus` 双定位记录，保留 {qc['n']:,} 条用于十分类。原标签 `Plastid` 按课程命名映射为 Chloroplast，但原注释的范围比叶绿体更宽。",
        f"训练集 {qc['audit']['train_n']:,} 条，测试集 {qc['audit']['test_n']:,} 条，严格沿用 header 的官方 `test` 标记。原论文按同源序列簇分组后划分数据 [1]；我们没有重新随机切分，也不根据测试分数挑选模型。完整序列摘要检查发现跨集合完全重复为 {qc['audit']['cross_split_identical_sequence_hashes']}。这项检查能发现重复序列，但不能替代完整的同源性审计。",
        '![完整数据集的类别与长度分布。少数类别的结果需要更谨慎解释。](results/images/fig0_data_qc.pdf){width=100% fig-pos="H"}',
        "## 四种方法有什么区别？",
        "| 方法 | 更新什么 | 具体设置 |\n|:--|:--|:--|\n| Frozen + kNN | 不更新 ESM | 平均表征 + 余弦距离 5-NN |\n| Linear probing | 只更新分类头 | 640 维输入，10 类 softmax |\n| LoRA | 低秩矩阵和分类头 | query/value，rank 8，alpha 16 |\n| Full fine-tuning | 编码器与分类头 | 不使用的 contact head 冻结 |",
        f"使用同一个 ESM-2 模型版本。各方法输入最多 {runs['frozen']['config']['max_length']} 个残基；较长序列保留首尾、删除中间，以尽量保留定位信号。表征平均时排除 padding 和特殊 token。三种训练方法均按预先设定的 {runs['lora']['config']['epochs']} 轮训练，使用最后 checkpoint；测试集只在训练结束后评估。",
        r"""LoRA 的做法是保留原权重 $W_0$，增加一个小的更新项：$W=W_0+(\alpha/r)BA$。训练时只更新 $A$、$B$ 和分类头，因此需要更新的参数少得多 [3]。

```{=latex}
\begin{algorithm}[H]
\caption{LoRA 训练流程}
\KwIn{官方训练集、预训练 ESM-2、rank $r=8$}
冻结原模型权重，初始化低秩矩阵 $A,B$ 和分类头\;
\For{每一轮训练}{
  \For{每组 mini-batch}{
    保留序列两端并分词，计算 ESM-2 表征\;
    对有效残基求平均，通过分类头得到预测\;
    累积交叉熵梯度，更新 $A,B$ 和分类头\;
  }
  保存 checkpoint\;
}
用最后 checkpoint 评估官方测试集\;
\end{algorithm}
```
""",
        "## 在什么机器上运行？",
        f"正式实验使用 **{hardware['gpu']}，{hardware['vram_gib']:.0f} GiB 显存**。Python {hardware['python'].split()[0]}，PyTorch {hardware['torch']}，bitsandbytes {hardware['bitsandbytes']}。LoRA 和 Full FT 使用 bf16、梯度检查点和 8-bit AdamW；Linear 的分类头使用普通 AdamW；Full FT 的实际 batch 为 {runs['full']['config']['batch_size']}，有效 batch 为 {runs['full']['effective_batch_size']}。所有样本、版本和参数均保存在 run.json 中。",
        "若显存不足，代码依次减小 batch、缩短输入、冻结前层；实际是否发生降级以运行记录为准。",
        f"本次 Full FT 的实际适配类型为 `{runs['full']['adaptation']}`，冻结前层数为 {runs['full']['config']['freeze_layers']}。",
        "# 实验流程",
        figure("fig6_workflow", "四条分支使用相同的官方测试集；统一评估与统计。", 85),
        "# 实验结果",
        "## 总体表现",
        "| 方法 | Accuracy | Macro F1 | ECE ↓ | 训练参数 | 时间/min | CUDA峰值/GiB |\n|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for s, row in table.iterrows():
        lines[-1] += (
            f"\n| {dict(frozen='Frozen + kNN', linear='Linear', lora='LoRA', full='Full FT')[s]} | {row.accuracy:.3f} | {row.macro_f1:.3f} | {row.ece:.3f} | {int(row.trainable_parameters):,} | {row.total_seconds / 60:.1f} | {row.gpu_peak_mb / 1024:.2f} |"
        )
    lines += [
        f"本次单次运行中，**{NAMES[best]} 的 Macro F1 最高，为 {bestrow.macro_f1:.3f}**；固定表征基线为 {base.macro_f1:.3f}。Macro F1 对每一类同等计分，因此能补充总体 accuracy 被大类别主导的问题。",
        "时间包含模型准备、训练、推断与保存；Linear 计入共享表征提取成本。显存为 PyTorch 记录的 CUDA 峰值分配量，不等同于 nvidia-smi 的全部设备占用。",
        figure(
            "fig1_parameter_efficiency",
            "四种方法需要更新的参数量。kNN 不更新参数，但仍需存储参考样本。",
        ),
        figure(
            "fig2_performance_cost",
            "性能与参数量、时间的关系；虚线仅连接不被其他点同时超越的方案。",
        ),
        f"**参数少不一定训练快。** LoRA 更新参数仅为 Full FT 的 {100 * a.trainable_parameters / f.trainable_parameters:.2f}%，峰值显存约为其 {100 * a.gpu_peak_mb / f.gpu_peak_mb:.0f}%，但二者均需通过 Transformer 计算梯度，本次总耗时分别为 {a.total_seconds / 60:.1f} 和 {f.total_seconds / 60:.1f} 分钟，几乎相同。Linear 连同表征提取只需 {l.total_seconds / 60:.1f} 分钟，是低预算下很有价值的起点。这里比较的是同一硬件、固定轮数的实测，不代表经过调参后的最优方案。",
        "## 哪些类别容易，哪些类别困难？",
        figure("fig8_class_f1", "每个定位类别的 F1；括号标出测试样本数。", 100),
        figure(
            "fig4_confusion_matrices",
            "Frozen 与 Full FT 的预测错误分布：数字是样本数，颜色按行归一化。",
            100,
        ),
        "类别图能区分“总体提高”和“每类都提高”。少数类别样本更少，分数容易波动；相近位置的混淆只能提示值得进一步检查的现象，不能直接证明某种生物机制。",
        f"Full FT 中，Chloroplast 的 F1 为 {per_class['Chloroplast']:.3f}，而 Lysosome/Vacuole 和 Peroxisome 分别为 {per_class['Lysosome/Vacuole']:.3f}、{per_class['Peroxisome']:.3f}；后两类测试样本只有 64 和 30 条。混淆矩阵还显示，508 条 Cytoplasm 中有 93 条被判为 Nucleus，808 条 Nucleus 中有 77 条被判为 Cytoplasm。这说明更高的总分仍不能消除类别不平衡与相近位置的区分难题。",
        "## 置信度与稳定性",
        figure(
            "fig3_calibration", "预测置信度与实际正确率的关系；越接近对角线越一致。", 85
        ),
        "ECE 衡量置信度与正确率的差距，越小越好。但低准确率模型也可能有较小 ECE，因此需要一起看分类表现；本实验没有用测试集调校准参数。",
        f"本次 Linear 的 ECE 为 {l.ece:.3f}，低于 LoRA 的 {a.ece:.3f} 和 Full FT 的 {f.ece:.3f}。Full FT 虽然分类更准，却更容易给错误预测很高的信心。因此，若需要根据置信度决定是否人工复核，应在独立验证数据上研究校准，不能直接把 softmax 当成可靠概率。Frozen 的 5-NN 投票还可能给真实类别零概率，导致 NLL 很大；评价脚本仅为数值稳定做概率裁剪。",
        figure(
            "fig9_robustness", "不同长度组的准确率，以及遮蔽序列两端后的变化。", 100
        ),
        "长度分组用于检查模型是否只擅长短序列。端点实验把两端各最多 10 个残基替换为 X，用来观察对端点信息的敏感性；这不是实际生物突变，也不能单独证明模型识别了定位信号。",
        f"Full FT 对不超过 128 残基的序列准确率为 {robust.loc[('full', 'length_0_128'), 'accuracy']:.1%}，对超过 512 残基的序列为 {robust.loc[('full', 'length_513_plus'), 'accuracy']:.1%}。长度组的类别构成也不同，不能把差异全部归因于截断。遮蔽端点后，其准确率从 {f.accuracy:.1%} 降为 {robust.loc[('full', 'terminal_X_mask_10_each'), 'accuracy']:.1%}，与端点信息有帮助的解释一致，但也可能包含输入分布变化的影响。",
        figure(
            "fig10_paired_intervals",
            "1,000 次配对 bootstrap 得到的 Macro F1 差值区间。",
            95,
        ),
        "每次重采样对四种方法使用相同测试 ID，并保持观测类别比例。区间跨过零表示这次重采样下差异方向不稳定。区间没有包含训练种子的变化，也不是同源家族独立重采样，不能把它当成所有不确定性的完整范围。",
        f"Full FT 相对 LoRA 的 Macro F1 提升为 {f.macro_f1 - a.macro_f1:.3f}，配对 95% CI 为 [{-pair.ci_high:.3f}, {-pair.ci_low:.3f}]，区间未跨零。相反，Linear 与 LoRA 的 Macro F1 差异区间跨零，因此这一次实验不足以稳定区分这两种方法的 Macro F1。这里列出的区间是逐项区间，未进行多重比较校正。",
        "## 表征与训练过程",
        figure(
            "fig5_embedding_pca", "只在训练表征上拟合 PCA，再展示测试蛋白的投影。", 95
        ),
        figure("fig7_training_curves", "训练 loss 和准确率随 epoch 的变化。", 95),
        "PCA 是高维表征的二维投影，只适合观察分布，不应凭图中是否成团来判断模型质量。训练曲线用于确认优化过程；最终泛化表现仍以独立测试集为准。",
        "# 讨论与总结",
        f"这次实验完成了从下载与清洗、四种适配，到独立测试、统计与作图的完整流程。按本次 Macro F1，{NAMES[best]} 表现最好；实际选择还应结合图中的时间、显存与少数类表现，而不是只看一个分数。",
        "需要保留三点限制：第一，只有一个训练种子，不能保证换一次训练仍有同样排名；第二，长序列截断和单标签设定简化了真实定位问题；第三，官方同源划分降低了监督数据泄漏风险，但不能排除 ESM 预训练语料中的重叠。后续最值得补充的是多个随机种子，以及带真实定位信号注释的分析。",
        "**数据与伦理声明：** 使用公开研究数据，引用原作者，原始序列不提交到 GitHub。本工作用于课程学习，不用于临床决策。代码与初稿由 AI 辅助，最终实验解释和提交内容由作者审核。",
        "# 参考文献",
        "1. Almagro Armenteros JJ, et al. DeepLoc: prediction of protein subcellular localization using deep learning. *Bioinformatics*. 2017;33:3387–3395. [doi:10.1093/bioinformatics/btx431](https://doi.org/10.1093/bioinformatics/btx431).",
        "2. Lin Z, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*. 2023;379:1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574).",
        "3. Hu EJ, et al. LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*. 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).",
    ]
    (ROOT / "lab8.qmd").write_text("\n\n".join(lines) + "\n")
    print(
        "Formal course report written; render with Quarto and visually review before delivery."
    )


if __name__ == "__main__":
    main()
