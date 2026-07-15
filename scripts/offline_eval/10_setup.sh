#!/usr/bin/env bash
# Fresh-box setup: venv + vllm + the five eval env packages (pull verifiers).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv-eval --python 3.12
source .venv-eval/bin/activate
uv pip install "vllm==0.22.0" "huggingface_hub[cli]"
uv pip install -e "rlm"            # rlms: local scaffold package (NOT PyPI)
uv pip install -e "rlm/training"   # rlm-train: envs depend on it
for e in oolong oolong_pairs browsecomp_plus longbench_codeqa longcot_mini; do
  uv pip install -e "rlm/training/environments/$e"
done
# longcot_mini pulls the pinned official `longcot` package (data + verifiers)
# from GitHub as a pip dep — needs network here, nothing at eval time.
python -c "import vllm, verifiers; print('vllm', vllm.__version__, '| verifiers', verifiers.__version__)"
echo "OK — activate with: source .venv-eval/bin/activate"
