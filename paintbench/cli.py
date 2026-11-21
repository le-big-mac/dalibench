"""The command line: estimate, attempt, render, judge, human-sheet, score, report.

Plus two helpers: `import` (bring pictures typed outside the harness into a
run) and `reference` (rebuild a painting's working copy from its original).

Rules:

- `attempt` and `judge` are the only commands that call a model. Each prints
  an estimate of what it is about to do, made with the jobs and flags it was
  given, and waits for confirmation. When the estimate holds nothing to make,
  the command says so and stops without asking.
- A run directory keeps one config, and nothing is written to it before the
  confirmation. `attempt` on a run that exists saves the config it was given
  with nothing dropped that the saved one lists (RunConfig.merged_into), and
  refuses a config that would change the conditions of attempts already on
  disk: they would be judged and scored together under one name. Changed
  conditions need a new run id. How the API is reached is not a condition
  (config.CONNECTION_OPTIONS), so the credentials, the address or a timeout
  may change while a run is carried on.
- A run keeps the paintings directory it began with, which its saved config
  names. `attempt` refuses a config that means another directory, and names
  both. `judge`, `human-sheet`, `score` and `report` read the run's own, and
  say so when the directory they would have picked without it is another.
  A run saved before its config named the directory is read from the
  default place, as it always was, and names that place the next time
  `attempt` or `judge --config` saves its config.
- `judge --config` takes from the other config its judges, and its `judge`,
  `retries` and `concurrency` sections where its file has them. Nothing else
  in the saved config is touched. A judge's name identifies its settings: a
  config that gives other settings to a judge that already has rankings is
  refused, and the changed judge needs a new name. The settings are those
  its calls are made with, so `judge.max_tokens`, which sets every judge's
  output limit, is one of them.
- An attempt that ended in error and a judging call that failed hold a model
  or a judge to nothing: no picture and no ranking came of them, and a
  corrected entry (a mistyped model id, say) is the usual reason to make
  them again.
- An attempt the provider blocked is run again only with `--retry-refused`:
  each is a request the platform flagged, so sending it again is a choice
  for whoever runs the benchmark, and is never made for them.
- One `attempt` or `judge` at a time works in a run directory. Each takes
  the run's lock (layout.run_lock) after the confirmation and before its
  first write, and holds it until its stage has ended and every worker has
  stopped. A second one says which process holds the run and exits with
  code 2, having made no call and changed no file.
- `--run` is a run directory, or a run id under ./runs (layout.resolve_run_dir).
- A failure that is expected (a config that does not load, a run that is not
  there, a refusal by a stage) is a message on stderr with no traceback, and
  exit code 2. Exit code 1 means the command did less than it was asked: it
  was not confirmed, or attempts were left unmade.
- An interrupt (SIGINT or SIGTERM) is exit code 130. What was in flight is
  finished and saved first, and "stopped" is said only when nothing is left
  running. A further signal while work is still finishing does not cut that
  short: it says how many are still in flight. `kill -9` abandons them.
- A record that cannot be read as JSON does not stop `score`, `report` or
  `render`: it is left out with one warning line that names it. `attempt`,
  `judge` and `human-sheet` stop and name the file, since they act on what
  is on disk.
- What is printed reaches a log file or a pipe line by line, as it happens.
- A page is served on the one address given with --host and never on every
  interface (human.check_host). The report server answers GET and HEAD for
  the files of the report directory and writes nothing.
"""

from __future__ import annotations

import argparse
import io
import os
import signal
import socket
import socketserver
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import yaml

from . import __version__
from .config import RunConfig
from .layout import STATE_ERROR, STATE_REFUSED, RunLayout, RunLocked, resolve_run_dir, run_lock
from .paintings import default_paintings_dir
from .types import ModelSettings
from .workers import ask_stop


def _one_line(text: Any) -> str:
    return " ".join(str(text).split())


def _saved_config(layout: RunLayout) -> RunConfig:
    try:
        cfg = RunConfig.from_file(layout.config_path)
    except (ValueError, yaml.YAMLError) as e:  # say which file: it is not one the command line named
        raise ValueError(f"{layout.config_path}: {_one_line(e)}") from None
    if "prompt_version" not in cfg.given:
        # Every saved config names its prompt version. One that does not was saved before there was a
        # choice, and any attempts in it by a model it lists were made with A1; saying so here keeps a
        # later save from mislabelling it. (A run that holds only imported pictures has no such attempts:
        # the key says nothing about them and holds the run to nothing.)
        cfg = replace(cfg, prompt_version="A1", given=cfg.given | {"prompt_version"})
    return cfg


def _load_run(args: argparse.Namespace, strict: bool = False) -> tuple[RunConfig, RunLayout]:
    """The run that --run names, with the config saved in it. `strict`: see RunLayout."""
    layout = RunLayout(resolve_run_dir(args.run), strict=strict)
    return _saved_config(layout), layout


def _say_whose_paintings(cfg: RunConfig, other: RunConfig | None = None) -> None:
    """Say that the run's own paintings are read, when the command would otherwise have picked another directory.

    That other directory is the one `other` means (the config given to `judge --config`), or the default
    place as it is found from here. A run whose saved config names no directory has nothing to say.
    """
    own = cfg.paintings_path
    picked = other.paintings_root if other is not None else default_paintings_dir().resolve()
    if own is not None and own.resolve() != picked:
        whose = "which the config given means" if other is not None else "the default place from here"
        print(f"Paintings: {own}, the directory this run was made with (and not {picked}, {whose}).")


def _other_paintings(cfg: RunConfig, saved: RunConfig) -> list[str]:
    """One line if `cfg` means another paintings directory than the one the run's saved config names.

    A saved config that names none (a run begun before one was always named) holds the run to nothing.
    """
    own = saved.paintings_path
    if own is None or own.resolve() == cfg.paintings_root:
        return []
    how = "" if cfg.paintings_dir else " (the default place from here, since this config names none)"
    return [f"the paintings directory is {str(cfg.paintings_root)!r} in this config{how} and was {str(own)!r} for the run"]


def _run_state(layout: RunLayout) -> tuple[bytes, tuple[tuple[str, int, int], ...]]:
    """What a judging plan is worked out from: the saved config, and every file of the attempts and the judgements
    by name, size and time of writing. Two readings that are equal mean nothing was written between them."""
    config = layout.config_path.read_bytes() if layout.config_path.is_file() else b""
    files = []
    for root in (layout.attempts_root, layout.judgements_root):
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file():
                    stat = path.stat()
                    files.append((path.relative_to(layout.root).as_posix(), stat.st_size, stat.st_mtime_ns))
    return config, tuple(sorted(files))


def _models_held(layout: RunLayout) -> set[str]:
    """The models with an attempt on disk that holds them to their settings: any that did not end in error."""
    return {a.model for a in layout.attempts() if a.state() != STATE_ERROR}


def _judges_held(layout: RunLayout) -> set[str]:
    """The judges with a ranking on disk, which holds them to their settings. A failed call holds none."""
    return {r.get("judge") for r in layout.judgements() if r.get("status") == "ok"}


def _with_judges(cfg: RunConfig, other: RunConfig, layout: RunLayout) -> RunConfig:
    """The run's config with what `judge --config` takes from the other one (see the module docstring)."""
    judges = {**{j.name: j for j in cfg.judges}, **{j.name: j for j in other.judges}}
    sections = {key: getattr(other, key) for key in ("judge", "retries", "concurrency") if key in other.given}
    merged = replace(cfg, judges=list(judges.values()), **sections)
    # Checked on the config as it would be saved: a judge's output limit can change through the `judge` section alone.
    changed = merged.changed_conditions(cfg, attempted=(), judged=_judges_held(layout))
    if changed:
        raise ValueError(
            f"this config would change a judge that already has rankings in {layout.root}: {'; '.join(changed)}. "
            "A judge's name identifies its settings, and rankings made under two settings would be averaged as one judge's. "
            "Give the changed judge a new name. Nothing was judged and the saved config is as it was."
        )
    return merged


# ---------------------------------------------------------------- stopping


def _waits_for_work(frame: Any) -> bool:
    """Whether a frame is a thread pool's own loop (concurrent.futures), in which a pool thread waits for work and runs it."""
    code = frame.f_code
    return code.co_name == "_worker" and code.co_filename.replace("\\", "/").endswith("concurrent/futures/thread.py")


def _say_from_handler(line: str) -> None:
    """Write one line to stderr from a signal handler, if it can be written.

    Not with print, where that can be helped. A handler runs wherever the main thread happens to be,
    and that can be inside an earlier print to the same stream: a signal that comes while the line for
    the one before it is being flushed. The buffered stream then raises RuntimeError ("reentrant
    call"), which took the place of the orderly stop and ended the command with a traceback (seen with
    two signals sent one straight after the other on a loaded machine). So when stderr is a stream of
    Python's own over a descriptor, the line goes to that descriptor in one write, past the buffer.
    Anything else put in the place of stderr (a capture, a tee) is written to in its own way, so the
    line goes where its other lines go. A line that cannot be written at that
    moment (the stream is in the middle of a write, or closed, or nobody is listening, or it is a
    stand-in that cannot do what print asks of it) is dropped. Nothing is ever raised from here: the
    handler runs wherever the main thread is, and an exception there is as harmful as the interrupt
    it is careful not to raise (workers.py).
    """
    stream = sys.stderr
    try:
        if isinstance(stream, io.TextIOWrapper):
            os.write(stream.fileno(), (line + "\n").encode(stream.encoding or "utf-8", "replace"))
            return
    except io.UnsupportedOperation:  # no descriptor under it: a capture in memory
        pass
    except Exception:  # nobody is listening any more, the stream is closed, its encoding is unknown
        return
    try:
        print(line, file=stream, flush=True)
    except Exception:  # whatever the stand-in cannot do (no flush, bytes only, closed, busy): the line is dropped
        pass


class _Stop:
    """How SIGINT and SIGTERM are taken while a command runs.

    The first is the stop. A stage turns it into an orderly one: it starts
    nothing more, and returns once what is in flight has finished and been
    saved. While a stage is waiting for its workers the handler does not raise
    the interrupt; it asks the stage to stop (workers.ask_stop), and the stage
    raises KeyboardInterrupt itself at its next checkpoint, a tenth of a second
    later at most. Raised from here it could land anywhere: inside the pool's
    own code while a thread is being started, or with a lock of a future held
    (workers.py says what each of those did). Outside a stage's wait the
    handler raises it, as Python's own does.

    A second KeyboardInterrupt would cut the wait for what is in flight short,
    and the command would say "stopped" and let go of the run while its
    workers went on calling models and writing files. So while work is in
    flight a further signal raises nothing. It says how many are still
    finishing, and that `kill -9` is the way to abandon them.

    SIGINT is set as well as SIGTERM because a command started in the
    background by a shell without job control inherits it ignored, and
    `kill -INT` would then do nothing.
    """

    def __init__(self) -> None:
        self.stopping = False  # the command is on its way out: a signal has asked it to stop, or its stage has ended
        self.making = ""  # what the stage in hand makes: "attempts" or "judging calls"
        self.before: set[threading.Thread] | None = None  # the threads alive before the stage began; None outside a stage

    def in_flight(self) -> list[threading.Thread]:
        """The worker threads of the stage in hand that are in the middle of an attempt or a judging call.

        A stage runs each one on a thread of a pool. The threads are those begun since the stage was
        entered whose stack holds the pool's loop; one whose innermost frame is that loop is waiting for
        work and is not counted.
        """
        if self.before is None:
            return []
        frames = sys._current_frames()
        busy = []
        for thread in threading.enumerate():
            frame = frames.get(thread.ident)
            if thread in self.before or frame is None or _waits_for_work(frame):
                continue
            while frame is not None and not _waits_for_work(frame):
                frame = frame.f_back
            if frame is not None:
                busy.append(thread)
        return busy

    def __call__(self, signum: int, frame: Any) -> None:
        left = len(self.in_flight())
        if self.stopping and left:
            _say_from_handler(
                f"paintbench: not stopped yet: {left} {self.making} still in flight are being finished and saved, and nothing "
                f"new is started. To abandon them instead, kill -9 {os.getpid()}; the same command then makes them again."
            )
            return
        self.stopping = True
        if ask_stop():  # a stage is waiting for its workers: it takes the stop at its next checkpoint
            return
        raise KeyboardInterrupt

    @contextmanager
    def stage(self, making: str) -> Iterator[None]:
        """Around a stage that works in threads: left only when every one of them has stopped.

        A stage waits for its own workers before it returns or raises. This is for the case that its
        wait was cut short all the same: the run's lock is not given up, and "stopped" is not said,
        while a worker is still calling a model or writing a file.
        """
        self.making, self.before = making, set(threading.enumerate())
        try:
            yield
        finally:
            self.stopping = True  # from here a signal has nothing to stop, and must not end the wait below
            while self.in_flight():  # looked at again and again: a pool thread outlives its work, so joining it could wait for ever
                time.sleep(0.05)
            self.before = None

    def install(self) -> dict[int, Any]:
        """Take SIGINT and SIGTERM for the command. Returns what to put back when it ends."""
        self.stopping = False
        previous: dict[int, Any] = {}
        for number in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[number] = signal.signal(number, self)
            except ValueError:  # not the main thread, so the signals are not ours to take
                break
        return previous


_stop = _Stop()


def _confirm(question: str, yes: bool, settings: list[ModelSettings]) -> bool:
    if yes or all(s.provider == "fake" for s in settings):
        return True
    if not sys.stdin.isatty():
        print("Not started: this would call model APIs. Run it again with --yes to confirm.")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def _summarise_states(results: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    return ", ".join(f"{n} {state}" for state, n in sorted(counts.items())) or "none made"


# ---------------------------------------------------------------- serving the report


class _ReportHandler(SimpleHTTPRequestHandler):
    """GET and HEAD for the files of one directory. No other method is answered and nothing is written."""

    server_version = "paintbench"
    sys_version = ""

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")  # a report built again must not be shown from a cache
        super().end_headers()

    def list_directory(self, path: Any) -> None:
        self.send_error(404, "File not found")  # the pages link to each other; a listing would add nothing


class ReportServer(ThreadingHTTPServer):
    """The report directory, read-only, on the one address given and not yet serving. Port 0 picks a free port."""

    def __init__(self, directory: Path | str, host: str, port: int):
        from .human import check_host

        host = check_host(host)
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        super().__init__((host, int(port)), partial(_ReportHandler, directory=str(directory)))

    def server_bind(self) -> None:
        from .human import check_host

        # Not HTTPServer.server_bind: it looks the host's name up in DNS, and nothing here needs that name.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
        check_host(self.server_name)  # a host name can resolve to every interface too


def _serve_report(directory: Path, host: str, port: int) -> None:
    try:
        server = ReportServer(directory, host, port)
    except (OSError, OverflowError) as e:  # OverflowError: a port number that is not one
        raise ValueError(
            f"cannot listen on {host} port {port}: {getattr(e, 'strerror', None) or e}. "
            "If another server holds the port, stop it or give another with --port."
        ) from None
    address, bound_port = server.server_address[:2]
    shown = f"[{address}]" if ":" in address else address
    print(f"Serving it at http://{shown}:{bound_port}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _check_serve(args: argparse.Namespace) -> None:
    """Refuse, before any work is done, a --serve with no address or with one that means every interface."""
    if not args.serve:
        return
    if not args.host:
        raise ValueError("--serve needs --host (the address to listen on, for example 127.0.0.1 or this machine's address on a private network)")
    from .human import check_host

    check_host(args.host)


# ---------------------------------------------------------------- commands


def cmd_estimate(args: argparse.Namespace) -> int:
    from .estimate import estimate, format_estimate

    cfg = RunConfig.from_file(args.config)
    cfg.check_attempts_can_be_made()
    print(format_estimate(estimate(cfg)))
    return 0


def _versions_on_disk(layout: RunLayout) -> set[str]:
    """The prompt versions that the run's attempts name in their stats.json (pictures that were imported name none)."""
    named = {(a.stats() or {}).get("prompt_version") for a in layout.attempts()}
    return {version for version in named if isinstance(version, str) and version != "imported"}


def _held_to_the_run(cfg: RunConfig, layout: RunLayout) -> tuple[RunConfig | None, RunConfig]:
    """(the run's saved config or None, the config to save) for `attempt` with `cfg`; ValueError if `cfg` would
    change the conditions of work already in the run.

    Two checks. Against the saved config: the conditions it holds the run to. Against the attempts on disk:
    the prompt version each one's stats.json names, whatever the saved config says, so that a run whose
    config was lost or edited, or whose attempts all ended in error, still gets no second wording. That
    holds for attempts this command would make again as well (--force, --retry-errors): they are replaced
    one at a time, and a command stopped half way would leave both wordings in one run.
    """
    saved = _saved_config(layout) if layout.config_path.is_file() else None
    to_save = cfg if saved is None else cfg.merged_into(saved)
    changed: list[str] = []
    if saved is not None:
        # Checked on the config as it would be saved, so that a section this file leaves out is compared as the run has it.
        changed = _other_paintings(cfg, saved) + to_save.changed_conditions(
            saved, attempted=_models_held(layout), judged=_judges_held(layout)
        )
    others = sorted(_versions_on_disk(layout) - {cfg.prompt_version})
    if others and not any(line.startswith("prompt_version ") for line in changed):
        changed.append(f"prompt_version is {cfg.prompt_version!r} in this config and attempts on disk were made with {', '.join(others)}")
    if changed:
        raise ValueError(
            f"this config would change the conditions of work already in {layout.root}: {'; '.join(changed)}. "
            "Attempts made under different conditions would be judged and scored together under one name. "
            "Use a new run_id for the changed conditions, or a new name for a changed model or judge. "
            "Nothing was run and the saved config is as it was."
        )
    return saved, to_save


def cmd_attempt(args: argparse.Namespace) -> int:
    from .estimate import estimate, format_estimate, to_do
    from .harness import run_attempts

    cfg = RunConfig.from_file(args.config)
    layout = RunLayout(cfg.run_dir, strict=True)
    jobs = cfg.attempt_jobs()
    if args.only:

        def matches(only: str, job: tuple) -> bool:
            return all(part in ("*", str(v)) for part, v in zip(only.split("/"), job))

        for only in args.only:  # each one must name something: a mistyped one is not passed over because another matched
            if len(only.split("/")) > 4:  # the parts past the fourth would be dropped, and the rest matched: more than was named
                raise ValueError(f"--only {only} has more than four parts (the form is painting/width/model/n, * for any)")
            if not any(matches(only, j) for j in jobs):
                raise ValueError(f"--only {only} matches no attempt in the config (the form is painting/width/model/n, * for any)")
        jobs = [j for j in jobs if any(matches(only, j) for only in args.only)]
    cfg.check_attempts_can_be_made()
    flags = {"retry_errors": args.retry_errors, "retry_refused": args.retry_refused}
    saved, to_save = _held_to_the_run(cfg, layout)
    est = estimate(cfg, layout, jobs=jobs, force_attempts=args.force, **flags)
    print(format_estimate(est, judging=False))
    print(f"\nRun directory: {layout.root}" + (" (carrying on; what this config adds is kept in its saved config)" if saved is not None else ""))
    count = to_do(est, judging=False)
    if not count:
        return 0
    models = [cfg.model(name) for name, row in est["models"].items() if row["attempts"]]
    if not _confirm(f"Run {count} attempts?", args.yes, models):
        return 1
    with run_lock(layout.root), _stop.stage("attempts"):
        # Checked again now that the run is held: another command may have added to it while the question was open.
        _held_to_the_run(cfg, layout)[1].save(layout.config_path)
        results = run_attempts(cfg, layout, jobs=jobs, force=args.force, **flags)
    print(f"\nAttempts: {_summarise_states(results)}.")
    if any(r["state"] == STATE_ERROR for r in results):
        print("An attempt that ended in error stays as it is; `attempt --retry-errors` runs those again.")
    if any(r["state"] == STATE_REFUSED and r.get("refusal_detected_by") == "provider" for r in results):
        print("An attempt the provider blocked stays as it is; `attempt --retry-refused` sends those requests again, if you choose to.")
    unmade = [j for j in jobs if not layout.attempt(*j).stats_path.is_file()]
    if unmade:
        # A model that was stopped, or an attempt that crashed outside its model calls.
        print(
            f"{len(unmade)} attempts were not made and nothing is on disk for them (see the STOP and CRASH lines above). "
            "Once the cause is dealt with, the same command makes them."
        )
        return 1
    print(f"Next: paintbench {'judge' if cfg.judges else 'report'} --run {layout.root}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from .layout import STATE_COMPLETE
    from .render import render_file

    layout = RunLayout(resolve_run_dir(args.run))
    count = 0
    for attempt in layout.attempts():
        if attempt.state() != STATE_COMPLETE or not attempt.picture_txt.is_file():
            continue
        if attempt.picture_png.is_file() and not args.force:
            continue
        render_file(attempt.picture_txt, attempt.picture_png)
        count += 1
    print(f"Rendered {count} pictures under {layout.attempts_root}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from .estimate import estimate, format_estimate, to_do
    from .judge import KINDS, run_judging

    cfg, layout = _load_run(args, strict=True)
    other = RunConfig.from_file(args.config) if args.config else None
    if other is not None:
        cfg = _with_judges(cfg, other, layout)
    if not cfg.judges:
        raise ValueError("no judges configured: add a `judges:` section to a config and pass it with --config")
    _say_whose_paintings(cfg, other)
    versions = sorted(_versions_on_disk(layout))
    if len(versions) > 1:
        raise ValueError(
            f"the attempts in {layout.root} were made with more than one wording of the prompt ({', '.join(versions)}). "
            "Pictures made under different wordings are not ranked together. Nothing was judged."
        )
    names = args.judge or None
    for name in names or []:
        cfg.judge_model(name)  # fail early on a wrong name
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip()) if args.kinds else KINDS
    shown = _run_state(layout)  # what the estimate below is worked out from
    est = estimate(cfg, layout, judge_names=names, kinds=kinds, force_judging=args.force, retry_failed=args.retry_failed, from_disk=True)
    print(format_estimate(est, attempts=False))
    count = to_do(est, attempts=False)
    if not count:
        return 0
    judges = [cfg.judge_model(name) for name, row in est["judges"].items() if any(row[kind] for kind in KINDS)]
    if not _confirm(f"Make {count} judging calls?", args.yes, judges):
        return 1
    with run_lock(layout.root), _stop.stage("judging calls"):
        if _run_state(layout) != shown:
            # Another command wrote to the run while the question was open: a judge added to its config, attempts
            # that change the sets, calls already made. What would be done now is not what was shown and agreed to.
            raise ValueError(
                f"{layout.root} changed while the question was open (another command wrote to it), so the calls to make are "
                "no longer the ones shown above. Nothing was judged and the saved config is as it was. Run the command again."
            )
        if other is not None:
            cfg.save(layout.config_path)  # so that score and report know the judges
        records = run_judging(
            cfg, layout, judge_names=names, kinds=kinds, force=args.force, seed=args.seed, retry_failed=args.retry_failed
        )
    failed = sum(1 for r in records if r.get("status") != "ok")
    print(f"\nJudging: {len(records)} calls made, {failed} failed.")
    if failed or any(row["already_failed"] for row in est["judges"].values()):
        print("A call that failed stays as it is; `judge --retry-failed` makes those again.")
    print(f"Next: paintbench score --run {layout.root} && paintbench report --run {layout.root}")
    return 0


def cmd_human_sheet(args: argparse.Namespace) -> int:
    from . import human
    from .paintings import load_painting

    cfg, layout = _load_run(args, strict=True)
    _check_serve(args)
    if args.record is not None:
        labels = human.parse_ranking(args.record)
        record = human.record_ranking(
            layout, args.painting, args.width, args.person, labels, notes=args.notes or "", sheet_id=args.sheet
        )
        print(f"Recorded {args.person}'s ranking of {len(record['ranking'])} pictures: {', '.join(record['ranking'])}")
        print(f"Saved to {layout.human_path(args.painting, args.width, args.person)}")
        return 0
    sheet = layout.human_sheet_path(args.painting, args.width, args.person)
    if not sheet.is_file() or args.new:
        _say_whose_paintings(cfg)
        painting = load_painting(args.painting, cfg.paintings_path)
        sheet = human.make_sheet(layout, painting, args.width, args.person, seed=args.seed, force=args.new)
        print(f"Blind sheet: {sheet}")
    else:
        human.check_sheet(layout, args.painting, args.width, args.person)  # out of date: refused, and --new is named
    if args.serve:
        human.serve_sheet(layout, args.painting, args.width, args.person, args.host, args.port)  # prints the address
        return 0
    print("Open it in a browser, put the pictures in order, then either:")
    print(f"  paintbench human-sheet --run {layout.root} --painting {args.painting} --width {args.width} --person {args.person} --record \"C, A, F, ...\"")
    print("or serve it so the Save button works:")
    print(f"  paintbench human-sheet --run {layout.root} --painting {args.painting} --width {args.width} --person {args.person} --serve --host <address>")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from . import score

    cfg, layout = _load_run(args)
    _say_whose_paintings(cfg)
    scores = score.compute_scores(layout, cfg)
    path = score.write_scores(layout, scores)
    print(score.format_scores(scores))
    print(f"\nScores saved to {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import report, score

    cfg, layout = _load_run(args)
    _check_serve(args)
    _say_whose_paintings(cfg)
    scores = score.compute_scores(layout, cfg)
    score.write_scores(layout, scores)
    index = report.build_report(layout, cfg, scores)
    print(f"Report: {index}")
    if args.serve:
        _serve_report(layout.report_dir, args.host, args.port)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from .importer import import_manifest

    run_dir = resolve_run_dir(args.run, new=True) if args.run else None
    layout = import_manifest(args.manifest, run_dir, args.paintings_dir, overwrite=args.overwrite)
    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))  # the import has checked it
    if not layout.config_path.is_file():
        root = layout.root.resolve()
        cfg = RunConfig.from_dict(
            {
                "run_id": root.name,
                "encoding": "palette",
                "paintings": [manifest["painting"]],
                "widths": [int(manifest["width"])],
                "runs_dir": str(root.parent),
                "paintings_dir": str(Path(args.paintings_dir).resolve()) if args.paintings_dir else None,
            }
        )
        cfg.save(layout.config_path)
    kinds = [ranking["kind"] for ranking in manifest.get("rankings") or []]
    print(f"Imported {len(manifest.get('attempts') or [])} pictures, {kinds.count('judge')} judge rankings and "
          f"{kinds.count('human')} human rankings into {layout.root}")
    print(f"Next: paintbench report --run {layout.root}")
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    from .paintings import grid_heights_for_ratio, load_painting
    from .reference import prepare_reference

    painting = load_painting(args.slug, args.paintings_dir)
    ref = painting.meta.get("reference") or {}
    if not ref.get("original_file"):
        raise ValueError(f"{painting.directory / 'meta.yaml'}: reference.original_file is not set")
    facts = prepare_reference(painting.directory / ref["original_file"], painting.reference_path, ref.get("crop_box"))
    facts["grid_heights"] = grid_heights_for_ratio(facts["height_over_width"])
    print(f"Working copy: {painting.reference_path} (a file that already held these pixels is left as it was)\n")
    print("Facts to record under `reference:` in meta.yaml (and `grid_heights:` for a new painting):")
    print(yaml.safe_dump(facts, sort_keys=False))
    for key in ("original_sha256", "reference_sha256", "reference_pixel_size"):
        if ref.get(key) and ref[key] != facts[key]:
            print(f"NOTE: {key} differs from meta.yaml ({ref[key]} recorded, {facts[key]} now)")
    return 0


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paintbench", description="DaliBench: The Persistence of Memory")
    parser.add_argument("--version", action="version", version=f"paintbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def run_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run", required=True, help="a run directory, or a run id under ./runs")

    def serve_args(p: argparse.ArgumentParser, what: str, port: int) -> None:
        p.add_argument("--serve", action="store_true", help=what)
        p.add_argument("--host", help="address to listen on with --serve (never all interfaces)")
        p.add_argument("--port", type=int, default=port, help=f"port to listen on with --serve (default {port})")

    p = sub.add_parser("estimate", help="count the calls a config would make and print an approximate cost")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("attempt", help="run contestant models and save transcripts and pictures")
    p.add_argument("--config", required=True)
    p.add_argument(
        "--only", action="append",
        help="limit to painting/width/model/n, with * for any (for example great-wave/32/*/1). Repeatable: an attempt is "
        "run if any one of them matches it",
    )
    p.add_argument("--yes", action="store_true", help="skip the confirmation after the estimate")
    p.add_argument("--force", action="store_true", help="run attempts again even if they are already on disk")
    p.add_argument("--retry-errors", action="store_true", help="run again the attempts whose saved state is error, and no other that is on disk")
    p.add_argument(
        "--retry-refused",
        action="store_true",
        help="run again the attempts the provider blocked, and no other that is on disk. Each is a request the platform "
        "flagged, so sending it again is your explicit choice; nothing does it unasked. An attempt a model declined in "
        "its own words is its answer and is not run again by this flag (only by --force). The record of the block is kept "
        "in the new stats.json under `earlier`",
    )
    p.set_defaults(func=cmd_attempt)

    p = sub.add_parser("render", help="turn picture.txt files into PNGs")
    run_arg(p)
    p.add_argument("--force", action="store_true", help="render again even if the PNG exists")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("judge", help="run the judge calls on saved attempts (never calls a contestant)")
    run_arg(p)
    p.add_argument(
        "--config",
        help="add this config's judges to the saved run, and take its judge, retries and concurrency sections where it has them; "
        "a judge that already has rankings keeps its settings (a changed judge needs a new name)",
    )
    p.add_argument("--judge", action="append", help="only this judge (repeatable); default: every judge configured")
    p.add_argument("--kinds", help="comma-separated subset of ranking,pairwise,peak")
    p.add_argument("--seed", type=int, help="seed for labels and order (default: random; each call's seed is saved)")
    p.add_argument("--yes", action="store_true", help="skip the confirmation after the estimate")
    p.add_argument("--force", action="store_true", help="judge again even if the judgement is already on disk")
    p.add_argument("--retry-failed", action="store_true", help="make again only the calls whose saved record is a failure")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("human-sheet", help="build a blind sheet for a person and record their ranking")
    run_arg(p)
    p.add_argument("--painting", required=True)
    p.add_argument("--width", required=True, type=int)
    p.add_argument("--person", required=True, help="a short name for the person ranking")
    p.add_argument("--record", help="record a ranking: the sheet's letters, best first, for example \"C, A, F, B\"")
    p.add_argument("--notes", help="notes to save with --record")
    p.add_argument("--sheet", help="with --record: the id of the sheet the ranking was made on; a ranking of a sheet since replaced is refused")
    serve_args(p, "serve the sheet so its Save button records the ranking", 8765)
    p.add_argument("--new", action="store_true", help="make a fresh sheet with new letters, discarding any ranking recorded for this person")
    p.add_argument("--seed", type=int, help="for tests only: a sheet made from a chosen seed has a key that can be worked out")
    p.set_defaults(func=cmd_human_sheet)

    p = sub.add_parser("score", help="compute rank scores, peak size, completion figures and judge-to-person agreement")
    run_arg(p)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("report", help="build the HTML page")
    run_arg(p)
    serve_args(p, "then serve the report directory, read-only, until interrupted", 8766)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("import", help="bring pictures typed outside the harness into a run directory")
    p.add_argument("manifest")
    p.add_argument("--run", help="a run directory, or a run id under ./runs (default: runs/<run_id from the manifest>)")
    p.add_argument("--paintings-dir")
    p.add_argument("--overwrite", action="store_true", help="replace attempts and rankings that are already in the run")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("reference", help="rebuild a painting's working copy (reference.png) from its original")
    p.add_argument("slug")
    p.add_argument("--paintings-dir")
    p.set_defaults(func=cmd_reference)

    return parser


def _fail(message: Any, code: int = 2) -> int:
    print(f"paintbench: {message}", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)  # a log file or a pipe then shows each line as it is printed
    except (AttributeError, ValueError):  # a stand-in for stdout that cannot be set up this way
        pass
    previous = _stop.install()
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        # A stage has waited for its workers by now (_Stop.stage), so nothing is left running.
        return _fail("stopped. What was finished is saved; the same command carries on from there.", 130)
    except yaml.YAMLError as e:
        return _fail(f"not valid YAML: {_one_line(e)}")
    except KeyError as e:
        # The package raises KeyError with a sentence for a name it does not know (a model, a judge, a grid
        # width). A bare key is a fault in the code, and keeps its traceback.
        if not (e.args and isinstance(e.args[0], str) and " " in e.args[0]):
            raise
        return _fail(e.args[0])
    except (OSError, ValueError, RunLocked) as e:
        return _fail(e)
    finally:
        for number, handler in previous.items():
            if handler is not None:
                signal.signal(number, handler)


if __name__ == "__main__":
    sys.exit(main())
