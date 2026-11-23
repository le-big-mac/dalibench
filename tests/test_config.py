"""The run config: the configs that come with the repository, and what a config may hold."""

from __future__ import annotations

import pytest

from paintbench.config import JudgeConfig, Limits, RunConfig
from paintbench.paintings import load_painting

from conftest import ROOT

MODEL = {"name": "m", "provider": "fake", "model": "s"}
BASE = {"run_id": "r", "paintings": ["great-wave"], "widths": [32], "models": [MODEL]}


@pytest.mark.parametrize("name", ["dalibench.yaml", "dalibench-256.yaml", "example.yaml", "offline-demo.yaml"])
def test_each_config_under_configs_loads_and_is_ready_to_run(name):
    cfg = RunConfig.from_file(ROOT / "configs" / name)
    assert cfg.run_dir == ROOT / "runs" / cfg.run_id  # the repository's runs/ directory, not configs/runs
    assert cfg.paintings_path is None  # the paintings beside the package: no path of one machine
    for painting in cfg.paintings:  # every painting has a grid at every width it is asked for
        heights = load_painting(painting, ROOT / "paintings").grid_heights
        assert all(width in heights for model in cfg.models for width in cfg.widths_for(model)), painting
    for entry in cfg.models + cfg.judges:  # an Anthropic key is named by its environment variable: nobody's own login
        assert entry.provider != "anthropic" or entry.options["auth"]["type"] == "api_key", entry.name


@pytest.mark.parametrize(
    "change, key",
    [
        ({"widhts": [16]}, "widhts"),
        ({"limits": {"max_calls": 9}}, "max_calls"),
        ({"judge": {"repeat": 5}}, "repeat"),
        ({"models": [{**MODEL, "effort": "max"}]}, "effort"),
    ],
)
def test_an_unknown_key_is_refused_and_named(change, key):
    """A misspelt key left alone would run the benchmark under the default it was meant to replace."""
    with pytest.raises(ValueError, match=key):
        RunConfig.from_dict({**BASE, **change})


@pytest.mark.parametrize("options", [{"api_key_env": "pasted-key-placeholder"}, {"auth": {"type": "api_key", "env": "pasted-key-placeholder"}}])
def test_a_key_pasted_where_the_name_of_its_variable_belongs_is_refused_and_not_repeated(options):
    """A config is saved with every run, so it is refused before anything is written, in words that do not hold the key."""
    with pytest.raises(ValueError, match="name of an environment variable") as refused:
        RunConfig.from_dict({**BASE, "models": [{**MODEL, "options": options}]})
    assert "pasted-key-placeholder" not in str(refused.value)


def test_aside_is_read_kept_in_the_saved_config_and_names_only_models_of_the_run():
    cfg = RunConfig.from_dict({**BASE, "models": [MODEL, {**MODEL, "name": "apart"}], "aside": ["apart"]})
    assert cfg.aside == ["apart"] and RunConfig.from_dict(cfg.to_dict()).aside == ["apart"]
    assert RunConfig.from_dict(BASE).aside == []
    with pytest.raises(ValueError, match="nobody"):
        RunConfig.from_dict({**BASE, "aside": ["nobody"]})


def test_dalibench_yaml_holds_the_conditions_of_the_full_run():
    cfg = RunConfig.from_file(ROOT / "configs" / "dalibench.yaml")
    assert (cfg.run_id, cfg.encoding, cfg.prompt_version, cfg.repeats, cfg.strip) == ("dalibench", "palette", "A5", 1, 16)
    assert len(cfg.paintings) == 16 and cfg.widths == [16, 32, 64, 128] and len(cfg.attempt_jobs()) == 11 * 16 * 4
    assert cfg.limits == Limits(max_steps=40, max_minutes=180, max_nudges=3)
    highest = {"thinking": "adaptive", "effort": "max", "display": "summarized"}
    assert [(m.name, m.provider, m.model, m.reasoning, m.max_output_tokens) for m in cfg.models] == [
        ("fable-5.2", "anthropic", "claude-fable-5-2", highest, 128000),
        ("opus-5.5", "anthropic", "claude-opus-5-5", highest, 128000),
        ("sonnet-5.5", "anthropic", "claude-sonnet-5-5", highest, 128000),
        ("haiku-5.5", "anthropic", "claude-haiku-5-5", highest, 128000),
        ("fable-5", "anthropic", "claude-fable-5", highest, 128000),
        ("sonnet-5", "anthropic", "claude-sonnet-5", highest, 128000),
        ("haiku-5.5-default-effort", "anthropic", "claude-haiku-5-5", {}, 128000),
        ("gemini-4-carbon", "google", "gemini-4-carbon", {"effort": "high"}, 65536),
        ("gpt-6.2-astra", "openai", "gpt-6.2-astra", {"effort": "max"}, 128000),
        ("glm-5.3", "openai_compat", "z-ai/glm-5.3", {"effort": "max"}, 128000),
        ("inkling-2-small", "openai_compat", "thinkingmachines/inkling-2-small", {"effort": "max"}, 128000),
    ]
    assert cfg.aside == ["haiku-5.5-default-effort"]
    assert [(j.name, j.provider, j.model, j.reasoning, j.max_output_tokens) for j in cfg.judges] == [
        ("fable-5.2-judge", "anthropic", "claude-fable-5-2", {"thinking": "adaptive", "effort": "high"}, 64000),
        ("opus-5.5-judge", "anthropic", "claude-opus-5-5", {"thinking": "adaptive", "effort": "high"}, 64000),
    ]
    assert cfg.judge == JudgeConfig(repeats=3, max_per_call=12, peak_size=True)  # a painting's pictures, up to eleven, in one ranking call
