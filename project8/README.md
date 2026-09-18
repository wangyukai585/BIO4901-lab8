# BIO4901 Lab 8：蛋白质定位预测

使用 ESM-2 150M，在 DeepLoc-1.0 官方划分上比较 Frozen + kNN、Linear probing、LoRA 和 Full fine-tuning。以下命令均在 `project8/` 执行。

## 安装环境

```bash
conda env create -f environment.yml
conda activate bio4901-lab8
```

Linux x86_64 / Python 3.11 也可使用完整依赖锁：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock-linux.txt
```

GPU 运行需要兼容 CUDA 的 NVIDIA 驱动。设备默认按 CUDA、MPS、CPU 顺序选择；可用 `device=cuda` 或单脚本 `--device` 指定。

## 下载与运行

```bash
python codes/01_download_data.py
snakemake -s scripts/Snakefile --cores 1 --forceall \
  --config subset=False biological=True device=cuda
```

`--forceall` 重新计算所有节点；去掉它可复用已有结果。全量配置使用完整官方训练/测试集、10 轮训练、512 残基输入。原始数据和模型缓存由脚本获取，不纳入 Git。

小规模检查：

```bash
snakemake -s scripts/Snakefile --cores 1 --config subset=True
```

使用 200 条训练、40 条测试序列；Linear/LoRA 两轮，Full FT 一个优化步骤。此模式只检查流程，不用于性能比较。

## 单独运行遮蔽对照

先完成训练，保留 `linear/last.pt`、`lora/last.pt`、`full/last.pt` 和 `data/embeddings_full/`：

```bash
python codes/11_biological_controls.py --results_dir results/full_run --device cuda
python codes/12_plot_biological_controls.py --results_dir results/full_run
```

分别遮蔽序列开头、结尾与内部等长片段，在相同测试 ID 上比较预测变化。

## 更新图表与报告

```bash
python codes/06_evaluate_and_bootstrap.py --output_dir results/full_run
python codes/07_make_figures.py --output_dir results/full_run --embedding_dir data/embeddings_full
python codes/09_extra_figures.py --output_dir results/full_run
python codes/12_plot_biological_controls.py --results_dir results/full_run
python codes/10_write_course_report.py --results_dir results/full_run
quarto render lab8.qmd --to pdf
```

报告模板为 `report/template.qmd`。渲染需要 Quarto、XeLaTeX 和 `algorithm2e`。将课程提供的 `bio4901.sty` 放在项目根目录可自动加载。macOS 使用 Times New Roman / PingFang SC；Linux 可指定已安装的字体：

```bash
quarto render lab8.qmd -M mainfont='DejaVu Serif' -M CJKmainfont='Noto Serif CJK SC'
```

## 文件位置

| 路径 | 内容 |
|---|---|
| `codes/` | 数据处理、训练、评价和作图脚本 |
| `scripts/Snakefile` | 可执行工作流 |
| `environment.yml`、`requirements-lock-*.txt` | 环境版本 |
| `data_dictionary.md` | 字段、标签和指标说明 |
| `results/qc/` | 数据质量检查 |
| `results/full_run/` | 正式指标、逐样本预测和图 |
| `results/full_run/biological_controls/` | 序列遮蔽对照 |
| `results/` 下的其他结果 | 小规模检查输出 |
| `lab8.qmd`、`lab8.pdf` | 报告源文件与 PDF |

数据来源：[DeepLoc-1.0](https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta)。保留官方 `test` 标记，排除双定位记录；训练/测试数量为 11,085 / 2,773。固定模型 `facebook/esm2_t30_150M_UR50D`，revision `a695f6045e2e32885fa60af20c13cb35398ce30c`。

## 参数与恢复

单脚本支持 `--subset_size --epochs --batch_size --device --output_dir --max_length --seed`。不传 `--subset_size` 即使用全部数据。Transformer 训练支持 `--effective_batch_size --precision --optimizer --resume`；实际参数保存在各策略的 `run.json`。

```bash
python codes/04_train_lora.py --max_length 512 --batch_size 4 \
  --effective_batch_size 16 --epochs 10 --output_dir results/full_run/lora \
  --resume results/full_run/lora/last.pt
```

恢复时应保持数据与训练配置一致。权重不随仓库分发，新环境需重新训练生成。

## 检查与打包

```bash
python -m pytest tests -q
python scripts/package_submission.py --output ../BIO4901_project8_submission.tgz
```

打包脚本读取最新 Git 提交，只包含 `project8/` 的已提交文件；不包含原始数据、权重和虚拟环境。
