#!/usr/bin/env python3
"""Back up irreplaceable run artifacts to a private HF dataset repo.

Uploads the control arm's rollout dumps and the wandb rollout-table snapshots
(both arms) to oerdogan/erlm-run-artifacts (private). Token from rlm/.env.
Run from EfficientRLM/outputs.
"""
import os, sys
from pathlib import Path

env = Path("/scratch/omeerdogan23/erlm/rlm/.env")
for line in env.read_text().splitlines():
    if line.startswith(("HF_TOKEN=", "HUGGINGFACE_TOKEN=", "HUGGING_FACE_HUB_TOKEN=")):
        os.environ.setdefault("HF_TOKEN", line.split("=", 1)[1].strip())

from huggingface_hub import HfApi

api = HfApi()
REPO = "oerdogan/erlm-run-artifacts"
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
for f in sys.argv[1:]:
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f),
                    repo_id=REPO, repo_type="dataset")
    print("uploaded", f)
print(f"backup repo: https://huggingface.co/datasets/{REPO}")
