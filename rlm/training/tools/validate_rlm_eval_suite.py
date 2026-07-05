#!/usr/bin/env python3
"""Validate the RLM eval-suite environments (EfficientRLM port).

Adapted from the ERLM-main validator; two changes. Assertions match the current
clean-pattern prologues (no plan hints anywhere; browsecomp_plus exports its
prologue from description.py). Prologues are pinned by full SHA-256 in addition
to readable substring diagnostics, so ANY silent edit to a user prologue fails
the gate: changing a prologue requires updating PROLOGUE_SHA256 here in the
same commit, which is exactly the loud, reviewable drift this tool exists to
force. Config wiring is checked in reverse: every env id referenced by any
config in training/configs must resolve to a validated package. No GPU or
network needed; run before every launch.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "environments"
CONFIGS = ROOT / "configs"

PACKAGES = {
    "oolong": "oolong",
    "oolong_pairs": "oolong_pairs",
    "browsecomp_plus": "browsecomp_plus",
    "longbench_codeqa": "longbench_codeqa",
}

# Full-text pins. A prologue edit MUST update the hash here in the same commit.
PROLOGUE_SHA256 = {
    "oolong": "08b4512dc465a55080f65e023c30fe49ec802b26f13202edf71c2a1cd0eca99c",
    "oolong_pairs": "7b552f0c47d2591f8523bd634dc00e999aa472f83fe70b24148a60e164683246",
    "browsecomp_plus": "7fc18d5eef6e5ceff65a230b88d7384ddbc5aef5f82ee985b7af63810f2ec5f6",
    "longbench_codeqa": "d44b8ae73ebd3b6d50dfe5219403a95a93f7212cd04c92b874877a6c417a1671",
}

sys.path[:0] = [str(ENVS / pkg) for pkg in PACKAGES.values()]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from browsecomp_plus.description import user_prologue as bcp_user_prologue  # noqa: E402
from browsecomp_plus.env import load_environment as load_bcp  # noqa: E402
from longbench_codeqa.env import load_environment as load_lbv2  # noqa: E402
from longbench_codeqa.env import user_prologue as lbv2_user_prologue  # noqa: E402
from oolong.env import load_environment as load_oolong  # noqa: E402
from oolong.env import user_prologue as oolong_user_prologue  # noqa: E402
from oolong_pairs.env import load_environment as load_pairs  # noqa: E402
from oolong_pairs.env import user_prologue as pairs_user_prologue  # noqa: E402

PROLOGUES = {
    "oolong": oolong_user_prologue,
    "oolong_pairs": pairs_user_prologue,
    "browsecomp_plus": bcp_user_prologue,
    "longbench_codeqa": lbv2_user_prologue,
}
LOADERS = {
    "oolong": load_oolong,
    "oolong_pairs": load_pairs,
    "browsecomp_plus": load_bcp,
    "longbench_codeqa": load_lbv2,
}


def _structure_checks() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for env_id, pkg in PACKAGES.items():
        base = ENVS / pkg
        pyproject = base / "pyproject.toml"
        checks[f"{pkg}_dir_exists"] = base.is_dir()
        checks[f"{pkg}_env_module_exists"] = (base / pkg / "env.py").exists()
        checks[f"{pkg}_init_exists"] = (base / pkg / "__init__.py").exists()
        checks[f"{pkg}_readme_exists"] = (base / "README.md").exists()
        if pyproject.exists():
            cfg = tomllib.loads(pyproject.read_text())
            eps = cfg.get("project", {}).get("entry-points", {}).get("verifiers.environments", {})
            checks[f"{pkg}_entrypoint_is_load_environment"] = (
                eps.get(env_id) == f"{pkg}:load_environment"
            )
        else:
            checks[f"{pkg}_entrypoint_is_load_environment"] = False
    return checks


def _config_env_ids() -> set[str]:
    ids: set[str] = set()
    for toml_path in sorted(CONFIGS.glob("*.toml")):
        cfg = tomllib.loads(toml_path.read_text())
        orch = cfg.get("orchestrator", {})
        for section in ("train", "eval"):
            for env in orch.get(section, {}).get("env", []):
                if "id" in env:
                    ids.add(env["id"])
    return ids


def main() -> int:
    checks = _structure_checks()

    # Full-text prologue pins + clean-pattern diagnostics.
    for name, text in PROLOGUES.items():
        checks[f"{name}_prologue_sha256_pinned"] = (
            hashlib.sha256(text.encode()).hexdigest() == PROLOGUE_SHA256[name]
        )
        checks[f"{name}_no_plan_hint"] = "Plan before you act" not in text
        checks[f"{name}_mentions_context_var"] = "`context`" in text
        checks[f"{name}_has_finalize_protocol"] = 'answer["ready"]' in text

    # Env-specific markers (task-definition facts live in the question
    # instruction under the clean pattern; answer-format mechanics in the prologue).
    import longbench_codeqa.env as _lbv2_env

    checks["oolong_pairs_mentions_pair_answer_format"] = "(id1, id2)" in pairs_user_prologue
    checks["bcp_clean_answer_instruction"] = "succinct answer" in bcp_user_prologue
    checks["bcp_mentions_provenance_header"] = "[BrowseComp+ doc" in bcp_user_prologue
    checks["lbv2_letter_format_in_question_instruction"] = (
        "A, B, C, or D" in getattr(_lbv2_env, "_QUESTION_INSTRUCTION", "")
    )

    # Signatures: config keys must land as kwargs, not vanish.
    oolong_params = inspect.signature(load_oolong).parameters
    checks["oolong_signature_has_min_ctx"] = "min_ctx" in oolong_params
    checks["oolong_signature_has_max_ctx"] = "max_ctx" in oolong_params
    checks["oolong_signature_has_exclude_numeric"] = "exclude_numeric" in oolong_params
    for name, loader in LOADERS.items():
        checks[f"{name}_signature_has_user_prologue"] = (
            "user_prologue" in inspect.signature(loader).parameters
        )

    # Reverse config wiring: every env id any config references must be validated here.
    unknown = _config_env_ids() - set(PACKAGES)
    checks["all_config_env_ids_are_validated_packages"] = not unknown

    for k, v in checks.items():
        print(f"{k}: {v}")
    if not all(checks.values()):
        if unknown:
            print(f"FAILED (config env ids with no validated package: {sorted(unknown)})")
        else:
            print("FAILED")
        return 1
    print("PASSED: structure, prologue hashes, signatures, and config wiring validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
