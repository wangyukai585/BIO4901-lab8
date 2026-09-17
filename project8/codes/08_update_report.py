#!/usr/bin/env python
"""Refresh report include from a chosen run; QMD structure never changes."""
from utils import *
def main():
    p=argparse.ArgumentParser(); p.add_argument('--results_dir',type=Path,default=ROOT/'results'); a=p.parse_args()
    df=pd.read_csv(a.results_dir/'metrics_table.csv'); qc=json.loads((ROOT/'results/qc/summary.json').read_text())
    smoke=(df.status!='FULL_RUN').any() or df.limited_steps.any()
    lines=['**[PLACEHOLDER - 待服务器全量运行后替换]**' if smoke else '**完整数据运行：请审核日志、显存降级及多随机种子复现后解释结果。**','',f"全量数据 QC：原始 {qc.get('raw_n',14004):,} 条；十类任务保留 {qc['n']:,} 条；排除双定位 {qc.get('excluded_dual_localization',146)} 条。训练 {qc['audit']['train_n']:,} 条，测试 {qc['audit']['test_n']:,} 条。跨集合完全重复序列数为 {qc['audit']['cross_split_identical_sequence_hashes']}。",'', '| 方法 | Accuracy | Macro F1 | ECE | 可训练参数 | 时间/s |','|:--|--:|--:|--:|--:|--:|']
    for row in df.itertuples(): lines.append(f'| {row.strategy} | {row.accuracy:.3f} | {row.macro_f1:.3f} | {row.ece:.3f} | {row.trainable_parameters:,} | {row.total_seconds:.1f} |')
    lines+=['',f'每种方法测试样本数：{int(df.test_n.iloc[0])}。时间包含该阶段模型准备、训练与推断；Linear 计入共享 embedding 提取成本。网络缓存、硬件与测试规模影响时间，不用于跨设备成本推断。','', '本地 MPS 内存不是 CUDA 显存；CUDA peak 字段留空。Full FT 冒烟仅执行一个 optimizer step，分数不能用于比较适配策略。' if smoke else '请检查 run.json 的 adaptation 字段；若为 partial_ft，应明确报告为部分层微调。','']
    for name,title in [('fig1_parameter_efficiency','可训练参数量'),('fig2_performance_cost','性能与成本'),('fig3_calibration','校准曲线'),('fig4_confusion_matrices','混淆矩阵'),('fig5_embedding_pca','Embedding PCA'),('fig7_training_curves','训练曲线')]:
        rel=(a.results_dir/'images'/f'{name}.pdf').resolve().relative_to(ROOT)
        lines += [f'![{title}。'+('冒烟占位结果。' if smoke else '')+f']({rel.as_posix()})'+'{width=95% fig-pos="H"}','']
    ci=pd.read_csv(a.results_dir/'bootstrap_ci.csv'); lines+=['## 配对 bootstrap 区间','', '| 差值方向 | 指标 | 估计差 | 95% CI |','|:--|:--|--:|:--|']
    for r in ci[ci.kind=='paired_difference'].itertuples(): lines.append(f'| {r.comparison} | {r.metric} | {r.estimate:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] |')
    lines += ['','序列级分层配对 bootstrap 固定观测类别比例。样本相关性与训练随机性未包含在区间内；没有同源簇 ID 时，不能把这些区间解释为家族独立泛化的完整不确定性。','']
    (ROOT/'results/report_results.qmd').write_text('\n'.join(lines))
if __name__=='__main__': main()
