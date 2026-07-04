from __future__ import annotations

import tomllib
from pathlib import Path

CFG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "smoke-qwen3-30b-correctness-multienv-oolong-browsecomp-gpt5nano.toml"
)


def _cfg() -> dict:
    return tomllib.loads(CFG.read_text())


def _env_by_id(envs: list[dict], env_id: str) -> dict:
    matches = [e for e in envs if e.get("id") == env_id]
    assert len(matches) == 1
    return matches[0]


def test_multienv_config_core_shape():
    cfg = _cfg()
    orch = cfg["orchestrator"]

    assert cfg["max_steps"] == 20
    assert cfg["seq_len"] == 8192
    assert orch["batch_size"] == 32
    assert orch["group_size"] == 4
    assert orch["train"]["sampling"]["extra_body"] == {"enable_thinking": False}
    assert orch["eval"]["sampling"]["extra_body"] == {"enable_thinking": False}


def test_multienv_config_train_envs_are_correctness_only():
    envs = _cfg()["orchestrator"]["train"]["env"]
    assert {e["id"] for e in envs} == {"oolong", "browsecomp_plus"}

    oolong = _env_by_id(envs, "oolong")
    assert oolong["ratio"] == 1.0
    assert "group_size" not in oolong  # inherit top-level group_size=4
    assert oolong["args"]["dataset_name"] == "spam"
    assert oolong["args"]["reward_style"] == "correctness"
    assert oolong["args"]["shaping_coef"] == 0.0

    bcp = _env_by_id(envs, "browsecomp_plus")
    assert bcp["ratio"] == 1.0
    assert "group_size" not in bcp  # inherit top-level group_size=4
    assert bcp["args"]["num_examples"] == 125
    assert bcp["args"]["start_index"] == 0
    assert bcp["args"]["k"] == 50
    assert bcp["args"]["reward_mode"] == "judge"
    assert bcp["args"]["judge_model"] == "openai/gpt-5-nano"
    assert bcp["args"]["reward_style"] == "correctness"
    assert bcp["args"]["shaping_coef"] == 0.0


def test_multienv_config_eval_split_is_disjoint_from_train_split():
    envs = _cfg()["orchestrator"]["eval"]["env"]
    assert {e["id"] for e in envs} == {"oolong", "browsecomp_plus"}

    oolong = _env_by_id(envs, "oolong")
    assert oolong["group_size"] == 1
    assert oolong["args"]["dataset_name"] == "trec_coarse"

    bcp = _env_by_id(envs, "browsecomp_plus")
    assert bcp["group_size"] == 1
    assert bcp["args"]["num_examples"] == 25
    assert bcp["args"]["start_index"] == 125
    assert bcp["args"]["k"] == 50
    assert bcp["args"]["reward_mode"] == "judge"
    assert bcp["args"]["judge_model"] == "openai/gpt-5-nano"
    assert bcp["args"]["reward_style"] == "correctness"
    assert bcp["args"]["shaping_coef"] == 0.0
