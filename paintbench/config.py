"""Run configuration.

A run is described by one YAML file. Adding a model, a judge or a painting is
an edit to that file. Every key, and what it may hold:

    run_id          required; a name (letters, digits, '.', '-' and '_')
    encoding        palette (the default) or value
    prompt_version  the version of the contestant prompt's wording; leave it out for the current one. A run records
                    it, and attempts are made only with the current wording. That wording tells the model it is
                    called again after each tool call, so attempts need limits.max_steps of 2 or more (both are
                    checked when attempts are to be made, not when a config is read)
    paintings       a list of painting slugs, none twice
    widths          a list of whole numbers of 1 or more, none twice
    repeats         attempts per model, painting and width: a whole number of 1 or more (default 1)
    strip_rows      rows per tool call: a whole number of 1 or more; left out, 16 for palette and 8 for value
    limits          max_steps     model calls per attempt: a whole number of 1 or more (default 40)
                    max_minutes   wall time per attempt: a number above 0 (default 60)
                    max_nudges    a whole number from 0 to 3 (default 3; the task allows at most three)
    retries         call_retries, restarts                  whole numbers of 0 or more (defaults 8 and 2)
                    backoff_seconds, backoff_max_seconds    numbers of 0 or more (defaults 2 and 60)
    concurrency     calls at once per provider: {provider or "default": a whole number of 1 or more}
                    (4 where nothing is said)
    models, judges  lists of entries, each a mapping with name, provider and model, and optionally reasoning,
                    max_output_tokens, price, options, widths, family, estimate and concurrency
                    (types.ModelSettings); names are safe as directory names and none is used twice
    judge           repeats (1 or more), max_per_call (2 to 26), pairwise_repeats, pairs_per_attempt and
                    max_tokens (1 or more, or left out), peak_size (true or false)
    paintings_dir   where the paintings are; left out, the default place (paintings.default_paintings_dir).
                    The config a run saves always names the directory, so the run keeps the one it began with
    runs_dir        where runs are kept (default "runs")

Rules:

- A relative `paintings_dir` or `runs_dir` is relative to the config file, not
  to the directory the command is run from.
- Every value is checked when the config is read. A value of the wrong type or
  out of range is a ValueError that names the key and says what is allowed;
  nothing is converted quietly (the text "3" is not the number 3).
- A failed API call is retried `call_retries` times. The wait starts at
  `backoff_seconds` and doubles up to `backoff_max_seconds`: by default 2, 4,
  8, 16, 32, 60, 60 and 60 seconds, about four minutes for one call before
  the conversation is given up.
- A run directory keeps the config it was made with. When `attempt` is given
  an edited config for a run that exists, `merged_into` gives the config to
  save: the edited one, with nothing the saved one lists dropped. Called on
  that config, `changed_conditions` says what it would change for the work
  already on disk, and such a config is refused.
- What is held for a model with attempts on disk, or a judge with rankings,
  is what decides the content of a request to it: `provider`, `model`,
  `reasoning`, `max_output_tokens` (for a judge, the limit in force: see
  `judge_settings`) and every key of `options` but the ones below. For a
  model, what decides the task is held as well: `encoding`, `prompt_version`,
  `strip_rows` and `limits`.
- What may change under the same name is what says how the API is reached and
  how long a reply is waited for, none of which is in the request the model
  answers. These keys of `options` (CONNECTION_OPTIONS), as the four adapters
  document them:
      auth, api_key_env                 who is asking (anthropic; the others)
      base_url                          where the API is. The same model id at another address is taken
                                        for the same model: a proxy, a gateway, another port
      extra_headers                     headers sent beside the request (openai_compat)
      timeout_seconds, sdk_max_retries  how long to wait and how often the SDK tries again
      poll_seconds, max_wait_seconds    the wait for a background reply (openai)
  and outside `options`: `price`, `concurrency`, `widths`, `family`,
  `estimate`, and the `retries` and `concurrency` sections. Each attempt's
  stats.json and each judgement record hold the settings they were made with,
  these included.
- Every other option is held, because it shapes the request or what is sent
  back to the model: `extra_body`, `max_tokens_param`, `effort_param`,
  `stream`, `stream_usage` and `replace_bad_tool_arguments` (openai_compat);
  `background`, `store`, `include_encrypted_reasoning` and `image_detail`
  (openai). An option this module does not know is held too.
- A run keeps one paintings directory: the grid heights and the reference
  images are conditions of its attempts and its rankings. `paintings_root`
  is the directory a config means, and the command line refuses to carry a
  run on with a config that means another.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from .adapters import REGISTRY
from .canvas import DEFAULT_STRIP_ROWS, ENCODINGS, PALETTE
from .prompts import PROMPT_VERSION
from .paintings import default_paintings_dir
from .types import ModelSettings

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# The settings of a model entry that decide what a call to it is, with every key of `options` that is not
# one of CONNECTION_OPTIONS. The rest (price, widths, family, estimate, concurrency) can change without the
# model's pictures or rankings meaning something else.
_CALL_SETTINGS = ("provider", "model", "reasoning", "max_output_tokens")

# The keys of `options` that say how the API is reached and how long a reply is waited for (see the module
# docstring). They are not in the request a model answers, so they may change while a run is carried on.
CONNECTION_OPTIONS = frozenset(
    {"auth", "api_key_env", "base_url", "extra_headers", "timeout_seconds", "sdk_max_retries", "poll_seconds", "max_wait_seconds"}
)


def check_name(kind: str, name: Any) -> str:
    """Names become directory names, so keep them to letters, digits, dot, dash and underscore."""
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"{kind} name {name!r} must be text of letters, digits, '.', '-' and '_' only, and not start with a symbol")
    return name


def _whole(key: str, value: Any, least: int = 1, most: int | None = None, optional: bool = False) -> Any:
    """A whole number in range, or a ValueError that names the key. True and 3.0 are not whole numbers."""
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < least or (most is not None and value > most):
        allowed = f"of {least} or more" if most is None else f"from {least} to {most}"
        raise ValueError(f"config: {key} must be a whole number {allowed}{' (or left out)' if optional else ''}, not {value!r}")
    return value


def _number(key: str, value: Any, above_zero: bool = False) -> Any:
    ok = not isinstance(value, bool) and isinstance(value, (int, float)) and 0 <= value < float("inf")
    if not ok or (above_zero and value == 0):
        raise ValueError(f"config: {key} must be a number {'above 0' if above_zero else 'of 0 or more'}, not {value!r}")
    return value


def _list(key: str, value: Any) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"config: {key} must be a list, not {value!r}")
    return value


def _mapping(key: str, value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config: {key} must be a mapping, not {value!r}")
    return value


def _no_repeats(key: str, values: list) -> list:
    if len(set(values)) != len(values):
        raise ValueError(f"config: {key} lists a value twice: {values}")
    return values


def _widths(key: str, value: Any) -> list[int]:
    return _no_repeats(key, [_whole(key, w) for w in _list(key, value)])


@dataclass
class Limits:
    max_steps: int = 40  # model calls per attempt
    max_minutes: float = 60.0  # wall time per attempt
    max_nudges: int = 3  # "the picture is not finished" messages per attempt; the task allows at most three

    def __post_init__(self) -> None:
        _whole("limits.max_steps", self.max_steps)
        _number("limits.max_minutes", self.max_minutes, above_zero=True)
        _whole("limits.max_nudges", self.max_nudges, 0, 3)


@dataclass
class Retries:
    call_retries: int = 8  # retries of one failed API call before the conversation is discarded
    restarts: int = 2  # fresh-conversation restarts after a discard before the attempt is an error
    backoff_seconds: float = 2.0  # first wait; doubles each retry
    backoff_max_seconds: float = 60.0

    def __post_init__(self) -> None:
        _whole("retries.call_retries", self.call_retries, 0)
        _whole("retries.restarts", self.restarts, 0)
        _number("retries.backoff_seconds", self.backoff_seconds)
        _number("retries.backoff_max_seconds", self.backoff_max_seconds)


@dataclass
class JudgeConfig:
    repeats: int = 3  # judging calls per set, each with fresh labels and order
    max_per_call: int = 8  # more attempts than this in a set switches to pairwise
    pairwise_repeats: int | None = None  # calls per pair; None means `repeats`
    pairs_per_attempt: int | None = None  # None: every pair. N: each attempt meets about N opponents
    peak_size: bool = True  # also rank each model's own sizes against each other
    max_tokens: int | None = None  # overrides the judge model's max_output_tokens when set

    def __post_init__(self) -> None:
        _whole("judge.repeats", self.repeats)
        _whole("judge.max_per_call", self.max_per_call, 2, 26)  # labels are single letters
        for key in ("pairwise_repeats", "pairs_per_attempt", "max_tokens"):
            _whole(f"judge.{key}", getattr(self, key), optional=True)
        if not isinstance(self.peak_size, bool):
            raise ValueError(f"config: judge.peak_size must be true or false, not {self.peak_size!r}")

    @property
    def effective_pairwise_repeats(self) -> int:
        return self.pairwise_repeats if self.pairwise_repeats else self.repeats


def _section(cls: type, data: Any, where: str) -> Any:
    data = _mapping(where, data)
    unknown = set(data) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"config: {where}: unknown keys {sorted(map(str, unknown))}")
    return cls(**data)


def _names_no_key(where: str, options: dict) -> None:
    """A config is saved with every run, so it may name the variable that holds an API key and never hold the key.

    Checked here, before anything is written, and the value is not echoed: it may be a key pasted in the wrong place.
    """
    auth = options.get("auth")
    named = {"options.api_key_env": options.get("api_key_env"), "options.auth.env": auth.get("env") if isinstance(auth, dict) else None}
    for key, value in named.items():
        if value is not None and not (isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)):
            raise ValueError(f"config: {where}: {key} must be the name of an environment variable, never the key itself")


def _entries(key: str, value: Any) -> list[ModelSettings]:
    """The `models` or `judges` list. ModelSettings checks the keys, the price and the entry's concurrency."""
    kind = key[:-1]
    entries = _list(key, value)
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"config: {key} must be a list of entries, each a mapping with name, provider and model, not {entry!r}")
        where = f"{kind} {entry.get('name', '?')!r}"
        for name in ("name", "provider", "model"):
            if isinstance(entry.get(name), (list, dict, bool)):
                raise ValueError(f"config: {where}: {name} must be text, not {entry[name]!r}")
        for name in ("reasoning", "options"):
            _mapping(f"{where}: {name}", entry.get(name))
        _names_no_key(where, entry.get("options") or {})
        for name, figure in _mapping(f"{where}: estimate", entry.get("estimate")).items():
            _number(f"{where}: estimate.{name}", figure)
        _whole(f"{where}: max_output_tokens", entry.get("max_output_tokens"), optional=True)
        _widths(f"{where}: widths", entry.get("widths"))
        if not isinstance(entry.get("family"), (str, type(None))):
            raise ValueError(f"config: {where}: family must be text, not {entry['family']!r}")
    settings = [ModelSettings.from_dict(entry) for entry in entries]
    names = [check_name(kind, m.name) for m in settings]
    if len(set(names)) != len(names):
        raise ValueError(f"config: {kind} names must be unique: {names}")
    return settings


def _union(old: list, new: list) -> list:
    return list(old) + [v for v in new if v not in old]


@dataclass
class RunConfig:
    run_id: str
    encoding: str = PALETTE
    prompt_version: str = PROMPT_VERSION  # which wording of the contestant prompt the run uses
    paintings: list[str] = field(default_factory=list)
    widths: list[int] = field(default_factory=list)
    repeats: int = 1
    models: list[ModelSettings] = field(default_factory=list)
    judges: list[ModelSettings] = field(default_factory=list)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    limits: Limits = field(default_factory=Limits)
    retries: Retries = field(default_factory=Retries)
    strip_rows: int | None = None  # None: the default for the encoding (16 palette, 8 value)
    # Entries kept apart: their attempts are made and judged like any other, but the scorer leaves their pictures
    # out of every comparison, so they have no rank score and the other entries' scores do not count them.
    aside: list[str] = field(default_factory=list)
    concurrency: dict[str, int] = field(default_factory=dict)  # per provider, plus "default"
    paintings_dir: str | None = None
    runs_dir: str = "runs"
    source_path: Path | None = None  # where the config was read from; relative paths resolve against it
    given: frozenset = frozenset()  # the keys the file set, so that a key it left out is not taken for a choice

    # ---- derived ----

    @property
    def strip(self) -> int:
        return self.strip_rows if self.strip_rows else DEFAULT_STRIP_ROWS[self.encoding]

    def concurrency_for(self, provider: str) -> int:
        return max(1, int(self.concurrency.get(provider, self.concurrency.get("default", 4))))

    def _resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        if p.is_absolute() or self.source_path is None:
            return p
        return (self.source_path.parent / p).resolve()

    @property
    def run_dir(self) -> Path:
        return self._resolve(self.runs_dir) / self.run_id

    @property
    def paintings_path(self) -> Path | None:
        """The paintings directory the config names. None if it names none: a config that no run has saved,
        or one saved before a run's config always named its directory."""
        return self._resolve(self.paintings_dir) if self.paintings_dir else None

    @property
    def paintings_root(self) -> Path:
        """The paintings directory this config means: the one it names, else the default place as it is
        found from here (the environment and the working directory decide: paintings.default_paintings_dir)."""
        return (self.paintings_path or default_paintings_dir()).resolve()

    def model(self, name: str) -> ModelSettings:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"no model named {name!r} in the config")

    def judge_model(self, name: str) -> ModelSettings:
        for m in self.judges:
            if m.name == name:
                return m
        raise KeyError(f"no judge named {name!r} in the config")

    def judge_settings(self, name: str) -> ModelSettings:
        """A judge's entry as its calls are made: `judge.max_tokens`, where set, in place of its own max_output_tokens."""
        judge = self.judge_model(name)
        return replace(judge, max_output_tokens=self.judge.max_tokens) if self.judge.max_tokens else judge

    def widths_for(self, model: ModelSettings) -> list[int]:
        return list(model.widths) if model.widths else list(self.widths)

    def attempt_jobs(self) -> list[tuple[str, int, str, int]]:
        """Every (painting, width, model name, attempt number) the config asks for."""
        jobs = []
        for painting in self.paintings:
            for model in self.models:
                for width in self.widths_for(model):
                    for n in range(1, self.repeats + 1):
                        jobs.append((painting, width, model.name, n))
        return jobs

    def check_attempts_can_be_made(self) -> None:
        """ValueError if attempts cannot be made with this config as it stands.

        Not checked when a config is read: a run saved under an older rule must stay readable to
        `judge`, `score` and `report`, which make no attempts.
        """
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"config: prompt_version is {self.prompt_version!r}, and attempts are made with {PROMPT_VERSION}, the wording in "
                "paintbench/prompts.py. A run made with another wording can still be judged, scored, and reported"
            )
        if self.limits.max_steps < 2:
            raise ValueError(
                "config: limits.max_steps must be 2 or more: the task message tells the model that it is called again "
                "after each write_rows call"
            )

    # ---- carrying on a run with an edited config ----

    def changed_conditions(self, saved: "RunConfig", attempted: Iterable[str], judged: Iterable[str] = ()) -> list[str]:
        """What this config would change for work already in a run whose saved config is `saved`.

        Call it on the config as it would be saved (`merged_into`, or what `judge --config` builds),
        so that a section the given file leaves out is compared as the run has it.

        `attempted` names every model that has attempts on disk, `judged` every judge that has
        judgements. One line per difference; an empty list means the run can carry on. Attempts made
        under different conditions would be judged and scored together under one model name, and
        rankings by a judge whose settings changed would be averaged as one judge's.

        The encoding, prompt version, strip size and limits are held once any attempt on disk is by a model the
        saved config lists: pictures that were only imported were not made under them.

        A model or a judge is held to what decides the content of a request to it, and not to how
        the API is reached (the module docstring gives both lists). A judge is compared as its
        calls are made (`judge_settings`), so a changed `judge.max_tokens` is a change to every
        judge that has rankings.
        """
        changed: list[str] = []

        def differ(what: str, was: Any, now: Any) -> None:
            if was != now:
                changed.append(f"{what} is {now!r} in this config and was {was!r} for the run")

        if set(attempted) & {m.name for m in saved.models}:
            differ("encoding", saved.encoding, self.encoding)
            differ("prompt_version", saved.prompt_version, self.prompt_version)
            differ("strip_rows", saved.strip, self.strip)
            for key in Limits.__dataclass_fields__:
                differ(f"limits.{key}", getattr(saved.limits, key), getattr(self.limits, key))
        limit_overridden = bool(saved.judge.max_tokens or self.judge.max_tokens)
        for kind, work, names, was, now in (
            ("model", "attempts", set(attempted), saved.models, self.models),
            ("judge", "judgements", set(judged), saved.judges, self.judges),
        ):
            listed = {m.name for m in was}
            for entry in now:
                if entry.name not in names:
                    continue
                if entry.name not in listed:
                    changed.append(f"{kind} {entry.name}: the run already holds {work} under this name that its saved config does not list (imported?)")
                    continue
                before = saved.model(entry.name) if kind == "model" else saved.judge_settings(entry.name)
                after = entry if kind == "model" else self.judge_settings(entry.name)
                for key in _CALL_SETTINGS:
                    # Name the key that was edited: a judge's limit can come from the `judge` section.
                    by_section = ", with judge.max_tokens applied," if kind == "judge" and key == "max_output_tokens" and limit_overridden else ""
                    differ(f"{kind} {entry.name}: {key}{by_section}", getattr(before, key), getattr(after, key))
                for key in sorted((set(before.options) | set(after.options)) - CONNECTION_OPTIONS):
                    differ(f"{kind} {entry.name}: options.{key}", before.options.get(key), after.options.get(key))
        return changed

    def merged_into(self, saved: "RunConfig") -> "RunConfig":
        """This config with nothing dropped that the run's saved config lists: what to save after carrying on.

        Paintings, widths, models and judges are the union, the saved ones first; repeats never go
        down. A model's widths are the union of what the two configs ask of it, so a model one of
        them leaves out is not asked for the other's widths. A model or judge named in both takes
        this config's entry (check `changed_conditions` on the result first). `judge`, `retries`,
        `concurrency` and `paintings_dir` stay as saved where this config's file left them out;
        encoding, prompt_version, strip_rows and limits are this config's, because they are what its attempts run
        under.
        """
        widths = _union(saved.widths, self.widths)
        was, now = {m.name: m for m in saved.models}, {m.name: m for m in self.models}
        models = []
        for name in _union(list(was), list(now)):
            asked = _union(saved.widths_for(was[name]) if name in was else [], self.widths_for(now[name]) if name in now else [])
            models.append(replace(now.get(name) or was[name], widths=None if set(asked) == set(widths) else asked))
        judges = {**{j.name: j for j in saved.judges}, **{j.name: j for j in self.judges}}
        own_paintings = "paintings_dir" in self.given and self.paintings_path is not None
        return replace(
            self,
            paintings=_union(saved.paintings, self.paintings),
            widths=widths,
            repeats=max(saved.repeats, self.repeats),
            models=models,
            judges=list(judges.values()),
            judge=self.judge if "judge" in self.given else saved.judge,
            retries=self.retries if "retries" in self.given else saved.retries,
            concurrency=dict(self.concurrency if "concurrency" in self.given else saved.concurrency),
            aside=list(self.aside if "aside" in self.given else saved.aside),
            paintings_dir=str(self.paintings_path) if own_paintings else (str(saved.paintings_path) if saved.paintings_path else None),
            given=self.given | saved.given,
        )

    # ---- reading and writing ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "encoding": self.encoding,
            "prompt_version": self.prompt_version,
            "paintings": list(self.paintings),
            "widths": list(self.widths),
            "repeats": self.repeats,
            "strip_rows": self.strip_rows,
            "limits": dict(self.limits.__dict__),
            "retries": dict(self.retries.__dict__),
            "concurrency": dict(self.concurrency),
            "models": [m.to_dict() for m in self.models],
            "judges": [m.to_dict() for m in self.judges],
            "judge": dict(self.judge.__dict__),
            "aside": list(self.aside),
            "paintings_dir": self.paintings_dir,
            "runs_dir": self.runs_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_path: Path | None = None) -> "RunConfig":
        if not isinstance(data, dict):
            raise ValueError("the config must be a mapping")
        known = {
            "run_id", "encoding", "prompt_version", "paintings", "widths", "repeats", "strip_rows", "limits", "retries",
            "concurrency", "models", "judges", "judge", "aside", "paintings_dir", "runs_dir",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"config: unknown keys {sorted(map(str, unknown))}")
        if not data.get("run_id"):
            raise ValueError("config: run_id is required")
        encoding = data.get("encoding", PALETTE)
        if encoding not in ENCODINGS:
            raise ValueError(f"config: encoding must be one of {ENCODINGS}, not {encoding!r}")
        prompt_version = data.get("prompt_version", PROMPT_VERSION)
        if not isinstance(prompt_version, str) or not prompt_version:
            raise ValueError(f"config: prompt_version must be the name of a wording, such as {PROMPT_VERSION!r}, not {prompt_version!r}")
        limits = _section(Limits, data.get("limits"), "limits")
        models, judges = _entries("models", data.get("models")), _entries("judges", data.get("judges"))
        providers = {"default", *REGISTRY, *(m.provider for m in models + judges)}
        concurrency = _mapping("concurrency", data.get("concurrency"))
        for provider, limit in concurrency.items():
            if provider not in providers:
                raise ValueError(f"config: concurrency names {provider!r}, which is not a provider; use one of {sorted(providers)}")
            _whole(f"concurrency.{provider}", limit)
        for key in ("paintings_dir", "runs_dir"):
            if not isinstance(data.get(key), (str, type(None))):
                raise ValueError(f"config: {key} must be a path as text, not {data[key]!r}")
        aside = _no_repeats("aside", _list("aside", data.get("aside")))
        for name in aside:
            if name not in {m.name for m in models}:
                raise ValueError(f"config: aside names {name!r}, which is not one of the models")
        return cls(
            run_id=check_name("run", str(data["run_id"])),
            encoding=encoding,
            prompt_version=prompt_version,
            paintings=_no_repeats("paintings", [check_name("painting", p) for p in _list("paintings", data.get("paintings"))]),
            widths=_widths("widths", data.get("widths")),
            repeats=_whole("repeats", data.get("repeats", 1)),
            models=models,
            judges=judges,
            aside=list(aside),
            judge=_section(JudgeConfig, data.get("judge"), "judge"),
            limits=limits,
            retries=_section(Retries, data.get("retries"), "retries"),
            strip_rows=_whole("strip_rows", data.get("strip_rows"), optional=True),
            concurrency=dict(concurrency),
            paintings_dir=data.get("paintings_dir") or None,
            runs_dir=data.get("runs_dir") or "runs",
            source_path=source_path,
            given=frozenset(data),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "RunConfig":
        path = Path(path).resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, source_path=path)

    def save(self, path: Path | str) -> None:
        """Write the resolved config into a run directory, with absolute paths so it stands alone.

        The paintings directory is always named: where the config names none, the default place as
        it is found now. A run is then held to the paintings it began with, wherever a later
        command is run from.
        """
        data = self.to_dict()
        data["paintings_dir"] = str(self.paintings_root)
        data["runs_dir"] = str(self.run_dir.parent)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Whole or not at all: written beside the file and moved into its place. Written straight into it, a stop
        # between the opening of the file and the writing left an empty config, and no command could read the run.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        os.replace(tmp, path)
