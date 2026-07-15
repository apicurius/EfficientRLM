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
    "longcot_mini": "longcot_mini",
}

# Full-text pins. A prologue edit MUST update the hash here in the same commit.
PROLOGUE_SHA256 = {
    "oolong": "c8c01cdb9989b30d2eb4245ccccc85fff1e896f7e4fe7d8395209b7f336a703a",
    "oolong_pairs": "5e052234e095f8d8e9e3915e3b8dc992f92cf9cdc074a2ed432061e85d982041",
    "browsecomp_plus": "e20ecc0c54b7013fbc77a1d27367dc6b44528e911411f021a71700f7bfcbd61f",
    "longbench_codeqa": "3a75cdbfb9fe5a96c90e2e7587aa7dc8a231ff365d6ddbe321f027f7d0c45509",
    "longcot_mini": "2f8eebd47ae415dfb66982b88c029d462f2df793fca436515b611d8004709f9a",
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

# longcot_mini is the only package whose module-level import needs a
# non-PyPI git dependency (`longcot`, see its pyproject.toml) that is not in
# training/uv.lock -- unlike the other four, it can be ABSENT from a locked
# rlm/training venv even though the package directory/files all exist. Import
# it defensively so that case is a loud FAILING check, not a crash that kills
# every other package's validation too.
try:
    from longcot_mini.env import load_environment as load_longcot_mini  # noqa: E402
    from longcot_mini.env import user_prologue as longcot_mini_user_prologue  # noqa: E402

    LONGCOT_MINI_IMPORTABLE = True
    LONGCOT_MINI_IMPORT_ERROR = ""
except ImportError as exc:
    load_longcot_mini = None
    longcot_mini_user_prologue = None
    LONGCOT_MINI_IMPORTABLE = False
    # `as exc` is deleted at the end of this block (exception-reference-cycle
    # avoidance) -- stash the message as a plain str for use later in main().
    LONGCOT_MINI_IMPORT_ERROR = str(exc)

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
if LONGCOT_MINI_IMPORTABLE:
    PROLOGUES["longcot_mini"] = longcot_mini_user_prologue
    LOADERS["longcot_mini"] = load_longcot_mini


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
    checks["longcot_mini_importable"] = LONGCOT_MINI_IMPORTABLE

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
    if LONGCOT_MINI_IMPORTABLE:
        checks["longcot_mini_mentions_solution_format"] = (
            "solution = " in longcot_mini_user_prologue
        )
        checks["longcot_mini_lifts_no_tools_rule"] = (
            "does NOT apply" in longcot_mini_user_prologue
        )

    # Signatures: config keys must land as kwargs, not vanish.
    oolong_params = inspect.signature(load_oolong).parameters
    checks["oolong_signature_has_min_ctx"] = "min_ctx" in oolong_params
    checks["oolong_signature_has_max_ctx"] = "max_ctx" in oolong_params
    checks["oolong_signature_has_exclude_numeric"] = "exclude_numeric" in oolong_params
    if LONGCOT_MINI_IMPORTABLE:
        lcm_params = inspect.signature(load_longcot_mini).parameters
        checks["longcot_mini_signature_has_difficulty"] = "difficulty" in lcm_params
        checks["longcot_mini_signature_has_start_index"] = "start_index" in lcm_params
        checks["longcot_mini_signature_has_enable_fallback"] = "enable_fallback" in lcm_params
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
        elif not LONGCOT_MINI_IMPORTABLE:
            print(f"FAILED (longcot_mini not importable: {LONGCOT_MINI_IMPORT_ERROR})")
            print(
                "  fix: uv pip install 'longcot @ git+https://github.com/LongHorizonReasoning/"
                "longcot.git@fb9649423f15f5b0091f8e988b100596cac592ca'"
            )
        else:
            print("FAILED")
        return 1
    print("PASSED: structure, prologue hashes, signatures, and config wiring validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
