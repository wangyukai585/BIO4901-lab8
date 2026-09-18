# BIO4901 Lab 8 — ESM-2 protein localization

**正式全量实验已于 2026-09-18 01:42（北京时间）完成，报告已更新为服务器真实结果。**

代码先通过本地小规模 subset 冒烟测试，再在 **RTX 3080 Ti 12 GiB** 上运行完整官方划分（11,085 train / 2,773 test）。三种训练策略均完成 10 epochs。正式结果位于 [`results/full_run/`](results/full_run/)，报告为 [`lab8.pdf`](lab8.pdf) 和 [`lab8.qmd`](lab8.qmd)。根 `results/metrics_table.csv` 与 `results/images/fig1*` 等保留本地冒烟记录，仍带占位标识，**不要将它们用于正式方法排名**；`results/images/fig0_data_qc.*` 是完整数据 QC。

| 策略 | Accuracy | Macro F1 | ECE ↓ | 总耗时/min | CUDA峰值/GiB |
|---|---:|---:|---:|---:|---:|
| Frozen + kNN | 0.693 | 0.574 | 0.063 | 6.9 | 0.77 |
| Linear probing | 0.753 | 0.590 | 0.043 | 6.5 | 0.77 |
| LoRA | 0.771 | 0.614 | 0.145 | 107.4 | 1.07 |
| Full fine-tuning | 0.793 | 0.685 | 0.153 | 107.8 | 2.12 |

Full FT 的 Macro F1 最高；Linear 的校准和成本更有优势。LoRA 更新参数只有 Full FT 的 0.42%，但本次训练耗时接近。Full FT 相对 LoRA 的 Macro F1 差值 0.071，配对 bootstrap 95% CI [0.045, 0.097]；区间只覆盖单次训练下的测试序列重采样不确定性。完整指标、配对区间、类别 F1、校准、长度与端点扰动图见正式结果目录。

CUDA bf16、AdamW8bit、batch 4、有效 batch 16、512 残基和 gradient checkpointing 均实测通过，未触发 OOM 降级。硬件和训练源代码版本见 `results/full_run/server_environment.json`；逐方法配置见 `run.json`。原始数据和模型权重不提交。

## 报告修订与补充实验（2026-09-18）

按课程 PPT 第 25/27/31/33 页逐项核对，详见 [requirements_audit.md](requirements_audit.md)。报告改为标准研究报告结构，以简短中文解释做法与结果，技术细节和辅助图移入附录。当前未提供 `bio4901.sty`；保留可选加载位置，不声称已套用课程样式。

补充了**开头 / 结尾 / 内部等长遮蔽**实验：对同一官方测试集、同一已训练模型，在截断后分别替换最多 10 个残基为 X；内部片段中心位于保留序列长度的 1/4 处，避免跨越长序列截断拼接点。按类别记录识别率和真实类别概率的变化。属于主 benchmark 后的解释性分析，不用于训练或选模型。

```bash
snakemake -s scripts/Snakefile --cores 1 --config subset=False biological=True
python codes/10_write_course_report.py --results_dir results/full_run
quarto render lab8.qmd --to pdf
```

**重新计算已有结果时须加 `--forceall`**；仓库已包含成品结果，不加会复用已有节点。

服务器已训练模型存在时，可仅运行 `11_biological_controls.py` 和 `12_plot_biological_controls.py`。新克隆仓库不含权重，须先运行训练节点生成 checkpoint。正式正文模板在 `report/template.qmd`，结果由报告脚本填入；新实验后仍需审阅结论。

仅打包本次 Lab 8 的已提交代码与结果（不含原始数据/权重）：

```bash
python scripts/package_submission.py --output ../BIO4901_project8_submission.tgz
```

课程最终要求合并两个项目提交，此压缩包只包含 `project8/`，不是替代另一项目的完整课程提交。

## 项目与数据

本仓库根目录内的 `project8/` 是课程提交项目。所有以下命令均在 `project8/` 执行。固定模型为 `facebook/esm2_t30_150M_UR50D`，不可变 revision `a695f6045e2e32885fa60af20c13cb35398ce30c`。

```text
project8/
  codes/01_download_data.py          # 下载、标签清洗、官方划分、全量 QC
  codes/02_extract_embeddings.py     # 固定 ESM + exact cosine 5-NN
  codes/03_train_linear_probe.py     # cached embedding + linear softmax
  codes/04_train_lora.py             # query/value rank-8 LoRA + head
  codes/05_train_full_ft.py           # full FT + CUDA preflight/OOM fallback
  codes/06_evaluate_and_bootstrap.py  # calibration, paired bootstrap, robustness
  codes/07_make_figures.py           # PDF/SVG/300dpi PNG
  codes/08_update_report.py          # 历史冒烟报告 include
  codes/09_extra_figures.py          # 类别 F1、鲁棒性、配对 CI
  codes/10_write_course_report.py    # 正式中文图文报告
  codes/utils.py, train_core.py, frozen_knn.py
  scripts/Snakefile
  tests/test_pipeline.py
  results/metrics_table.csv, bootstrap_ci.csv, images/, qc/
  environment.yml, requirements.txt, requirements-lock-macos.txt, requirements-lock-linux.txt
  command_log.txt, data_dictionary.md, lab8.qmd, lab8.pdf
```

官方下载共 14,004 条，**排除 146 条 Cytoplasm-Nucleus 双定位记录**，保留 13,858 条。`Plastid` 映射到课程的 Chloroplast（原注释含义更宽）。官方 train/test 为 **11,085 / 2,773**，不重新随机划分。原论文以同源簇分五折、留一折测试；序列 hash 审计跨集合完全重复为 0，但不能独立证明同源阈值，亦不能排除 ESM 预训练重叠。详情见 `results/qc/` 与 [原论文](https://doi.org/10.1093/bioinformatics/btx431)。训练集内部不创建随机验证集；固定 epoch、使用最后 checkpoint，不通过测试成绩调参。

## 环境搭建

```bash
conda env create -f environment.yml
conda activate bio4901-lab8
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
```

或 Python 3.11 venv：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

本次 Mac 环境位于仓库根的 `.venv/`，激活命令为 `source ../.venv/bin/activate`。版本在 environment.yml 精确固定；macOS 完整实际安装快照另存 lock 文件，**不要在 Linux 直接复用 macOS 的传递依赖快照**。`requirements-lock-linux.txt` 是为 Linux x86_64 / Python 3.11 解析的完整依赖锁（包括 CUDA 12.4 依赖），可在对应服务器 venv 中 `pip install -r requirements-lock-linux.txt`。已在本次 Linux CUDA 服务器安装并完成全量运行。bitsandbytes 0.45.5 仅在 Linux x86_64 安装；MPS/CPU 使用 torch AdamW。不要求 Mac 有 CUDA。

## 本地冒烟：一条命令

```bash
snakemake -s scripts/Snakefile --cores 1 --config subset=True
```

默认训练 200、测试 40，十类均至少保留一条，种子固定；采样只发生在原集合内部。Linear/LoRA 两轮，Full FT 一个 optimizer step、batch 1；最大残基数 128。全量 QC 始终使用全部数据。已有产物时 Snakemake 跳过已完成节点；完整复验加 `--forceall`。串行执行避免 GPU 资源竞争。

## CUDA 服务器全量复现（本次为 RTX 3080 Ti）

在服务器 clone 后，进入 `project8/` 并安装上面的环境。建议先确认 `nvidia-smi` 正常及当前 PyTorch wheel 能使用 CUDA：

```bash
python -c "import torch, bitsandbytes; print(torch.cuda.get_device_name(0)); print('bf16:', torch.cuda.is_bf16_supported())"
python codes/05_train_full_ft.py --device cuda --probe_only --batch_size 4 --effective_batch_size 16 --max_length 512 --output_dir results/server_probe
snakemake -s scripts/Snakefile --cores 1 --config subset=False
```

只改变 `subset=False` 即运行全量，输出隔离到 `results/full_run/`，embedding 在 `data/embeddings_full/`，不会覆盖 Mac 占位结果。默认 10 epochs；可附加 `epochs=20`，须在看测试分数前确定。默认最大长度 512，保留两端；单脚本可使用 `--max_length 1022`。不传 `--subset_size` 或传 `None` 即全部训练记录，`--eval_size` 不传时随 subset_size，二者均不传即全部测试记录。

若租用镜像的驱动不兼容环境默认 CUDA wheel，应按 PyTorch 的官方安装方式安装 **同版本 torch 2.6.0** 的兼容 CUDA wheel，并记录 `torch.__version__`；不在 Mac 上伪称验证过 CUDA。预检必须在目标 GPU 实机完成。

## 参数、内存与中断恢复

所有训练入口支持 `--subset_size --epochs --batch_size --device --output_dir --max_length --seed`。Transformer 训练还支持 `--effective_batch_size --precision {auto,bf16,fp32} --optimizer {adamw,adamw_8bit} --lora_rank --freeze_layers --resume --max_steps`。

- CUDA 默认 bf16（硬件检测不支持则 fp32）、gradient checkpointing、8-bit AdamW、batch 4 + accumulation 到有效 batch 16。MPS 运行时检测 bf16 autocast，否则 fp32。
- CUDA Full FT 自动先运行微型输入，再运行**配置最大 batch/长度并执行 optimizer step**。峰值包括 optimizer 状态；不是用微型 batch 简单线性外推，仍不是 OOM 保证。
- 探测 OOM 按 **batch 减半到 1 → length 减半到 min_length=128 → 冻结前 5/10/.../29 层** 重试。训练循环 OOM 从固定预训练基座重启当前策略并采用下一配置，进程替换释放全部 GPU 分配。失败重试不是继续部分更新的模型。
- 降级写入 `memory_probe.json`、`runtime_oom.json`、`run.json`；冻结层后方法标为 `partial_ft`，不同长度也会标记为预处理不一致。不能把它无说明地当 full FT 同条件比较。
- 每 30 秒心跳，tqdm ETA，固定 step 日志，同时写终端和 `<output_dir>/training.log`。epoch 完成后原子覆盖 `last.pt`；`last.json` 保留小配置。可恢复 optimizer、RNG 与下一 epoch，最多重做中断中的一个 epoch。
- `--max_steps` 是开发测试上限，不用于正式轮次续跑；中途截断的 epoch 保存为已完成的开发片段。

例如恢复相同目录、相同参数的 LoRA（除 epoch 总数可增加）：

```bash
python codes/04_train_lora.py --max_length 512 --batch_size 4 --effective_batch_size 16 --epochs 10 --output_dir results/full_run/lora --resume results/full_run/lora/last.pt
```

不要加载来源不可信的 `.pt` checkpoint。Frozen 的参考 embedding 保存在忽略的缓存中；kNN 本身无训练迭代。Linear 的 backbone 不参与训练，训练参数 6,410；LoRA 为 620,810；full classification 模型为 148,391,050（未使用的 contact head 不训练）。

## 评估、画图和更新报告

```bash
python codes/06_evaluate_and_bootstrap.py --output_dir results/full_run --n_bootstrap 1000
python codes/07_make_figures.py --output_dir results/full_run --embedding_dir data/embeddings_full
python codes/09_extra_figures.py --output_dir results/full_run
python codes/10_write_course_report.py --results_dir results/full_run
quarto render lab8.qmd --to pdf
```

正式报告生成器会拒绝不完整数据或冒烟运行，自动读取结果和图路径；重新训练后仍须检查分析文字与 PDF 排版。旧的 `08_update_report.py` 仅用于历史冒烟报告 include。使用 `bash scripts/run_server.sh` 可自动记录硬件、预检、运行完整 DAG 并生成附加图；报告最后在安装 Quarto/TeX 的机器渲染。

本地已使用 Quarto 1.7.32、XeLaTeX 渲染。`bio4901.sty` 在 YAML 的 `IfFileExists` 位置可选加载；用户放入 project8 根目录后重新 render。Linux 字体可用：`quarto render lab8.qmd -M mainfont='DejaVu Serif' -M CJKmainfont='Noto Serif CJK SC'`（先安装相应字体及 TeX/algorithm2e）。缺课程样式时不会冒充已应用课程模板。

CSV 包括固定十类 macro F1、accuracy、ECE、NLL、Brier、参数/成本，以及分组鲁棒性。bootstrap 是 1,000 次配对、按类别分层的序列重采样；未有簇 ID，不能当独立家族重采样。PCA 只在训练 embedding 拟合。合成鲁棒性输入为两端各最多 10 个残基替换 X，不能当真实生物突变。

## 验证与 Git

```bash
python -m pytest tests -q
snakemake -s scripts/Snakefile --cores 1 --config subset=False -n
```

已实测：MPS ESM bf16 forward/backward、Frozen、Linear/LoRA 两轮、Full FT 单 batch、checkpoint 参数恢复、梯度、完整 Snakemake smoke DAG、统计/图表与 PDF。CUDA/bitsandbytes 内核与预检已在 RTX 3080 Ti 实测通过；真实 OOM 未出现，降级顺序以注入异常的测试覆盖。

`.gitignore` 排除 raw/processed、embedding、权重、虚拟环境和工具安装；保留小 checkpoint JSON、CSV 与图。已按用户授权分模块提交并推送到 GitHub。若 Mac 系统 Git 提示 Xcode 许可，可在已安装 Command Line Tools 的本机使用 `DEVELOPER_DIR=/Library/Developer/CommandLineTools git ...`。推送前在仓库根检查 `git status`。
