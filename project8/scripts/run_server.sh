#!/usr/bin/env bash
# Detached full experiment launcher. Activate the intended Python environment first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/full_run
exec > >(tee -a results/full_run/launcher.log) 2>&1
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "FAILED exit=%s at %s\n" "$rc" "$(date -Iseconds)" > results/full_run/STATUS; fi' EXIT
printf 'RUNNING %s\n' "$(date -Iseconds)" > results/full_run/STATUS
python - <<'PY'
import json, platform, sys, torch, bitsandbytes, subprocess
from pathlib import Path
assert torch.cuda.is_available(), 'Server run requires CUDA'
info={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'bitsandbytes':bitsandbytes.__version__,'gpu':torch.cuda.get_device_name(0),'vram_gib':torch.cuda.get_device_properties(0).total_memory/2**30,'bf16':torch.cuda.is_bf16_supported()}
print(json.dumps(info,indent=2)); Path('results/full_run/server_environment.json').write_text(json.dumps(info,indent=2))
PY
python codes/05_train_full_ft.py --device cuda --probe_only --batch_size 4 --effective_batch_size 16 --max_length 512 --output_dir results/server_probe
snakemake -s scripts/Snakefile --cores 1 --config subset=False device=cuda --rerun-incomplete
printf 'COMPLETED %s\n' "$(date -Iseconds)" > results/full_run/STATUS
