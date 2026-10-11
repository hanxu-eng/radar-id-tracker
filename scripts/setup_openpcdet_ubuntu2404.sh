#!/usr/bin/env bash
# Only installs into an already activated Python 3.11 environment; no sudo.
set -euo pipefail
task_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_root"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This recipe targets Ubuntu 24.04 x86_64 with an NVIDIA discrete GPU.' >&2
  exit 2
fi
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 24.04 ]]; then
  echo 'Expected Ubuntu 24.04; use a separate reviewed recipe for other releases.' >&2
  exit 2
fi
python - <<'PY'
import os,sys
if sys.version_info[:2] != (3,11):
    raise SystemExit('Activate the Python 3.11 Conda environment first.')
if not os.environ.get('CONDA_PREFIX') and sys.prefix == sys.base_prefix:
    raise SystemExit('Use an isolated Conda/venv environment, not system Python.')
PY
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
command -v nvcc >/dev/null
nvcc --version | grep -q 'release 12.6' || { echo 'CUDA Toolkit 12.6 is required.' >&2; exit 2; }
command -v gcc >/dev/null
command -v g++ >/dev/null
command -v git >/dev/null
python -m pip install --upgrade 'pip>=24,<26'
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-openpcdet-ubuntu2404.txt
python -m pip install -e . --no-deps --no-build-isolation
# Neither datasets nor annotations are needed to download/compile the models.
python examples/prepare_openpcdet.py --download --models second_iou pv_rcnn
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  task_cuda_arch="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('GPU not visible; check the NVIDIA driver before building.')
major,minor=torch.cuda.get_device_capability()
print(f'{major}.{minor}')
PY
)"
  export TORCH_CUDA_ARCH_LIST="$task_cuda_arch"
fi
export MAX_JOBS="${MAX_JOBS:-2}"
(
  cd outputs/vendor/OpenPCDet
  python -m pip install -e . --no-deps --no-build-isolation
)
python -m pip check
python examples/check_openpcdet_runtime.py --output outputs/ubuntu2404/runtime_check.json
python -m pip freeze > outputs/ubuntu2404/pip-freeze.txt
echo 'Runtime checks passed. Save runtime_check.json and pip-freeze.txt with evaluation results.'
