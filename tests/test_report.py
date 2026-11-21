"""The report: the pages built from a run, what they show, and what they may not hold.

Each run here is made by the real attempt and judge stages with scripted models. No model is called.
"""

from __future__ import annotations

import base64
import io
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

from PIL import Image

from paintbench.canvas import load_picture
from paintbench.config import RunConfig
from paintbench.harness import run_attempts
from paintbench.judge import run_judging
from paintbench.layout import RunLayout, read_json, write_json
from paintbench.paintings import load_painting
from paintbench.render import render_picture
from paintbench.report import build_report

from conftest import ROOT

PAINTINGS = ROOT / "paintings"
SLUG, WIDTH = "great-wave", 16
ORDER = ["cy", "ada", "bob"]  # the judge's order, best first: not the order of the names
STANDINGS = [("cy", "1.00"), ("ada", "0.50"), ("bob", "0.00")]  # rank score: the share of its comparisons a model won


def scored_run(tmp_path: Path, **others: dict) -> tuple[RunConfig, RunLayout]:
    """The Great Wave typed by three scripted models and ranked by a scripted judge in ORDER.

    `others` are further models, by name, each with the options that say how its attempt ends.
    """
    script = {"provider": "fake", "model": "script"}
    models = [{"name": name, **script} for name in sorted(ORDER)]
    models += [{"name": name, **script, "options": options} for name, options in others.items()]
    cfg = RunConfig.from_dict({
        "run_id": "small",
        "paintings": [SLUG],
        "widths": [WIDTH],
        "paintings_dir": str(PAINTINGS),
        "runs_dir": str(tmp_path / "runs"),
        "models": models,
        "judges": [{"name": "judge", **script, "options": {"behaviour": "judge", "order": ORDER}}],
        "judge": {"repeats": 1},
    })
    layout = RunLayout(cfg.run_dir)
    run_attempts(cfg, layout, progress=lambda line: None)
    run_judging(cfg, layout, progress=lambda line: None, seed=1)
    return cfg, layout


def attempt_page(layout: RunLayout, model: str) -> Path:
    return layout.report_dir / "attempts" / SLUG / str(WIDTH) / model / "1.html"


def standings(index: str) -> list[tuple[str, str]]:
    """(model, rank score) for each row of the table the index page leads with, in the order shown."""
    return re.findall(r'<a href="#model-\d+">(\w+)</a></th><td class="n">([\d.]+)</td>', index)


def pictures(index: str) -> dict[str, Image.Image]:
    """{model: the picture on its card} for the picture cards of the index page, in the order shown."""
    card = rf'<li class="card" data-attempt="{SLUG}/{WIDTH}/(\w+)/1">.*?<img class="typed" src="data:image/png;base64,([^"]+)"'
    return {model: Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB") for model, data in re.findall(card, index, re.S)}


class Page(HTMLParser):
    """A built page: its tags, the address in every href and src, and the policy it gives the browser."""

    def __init__(self, path: Path):
        super().__init__()
        self.source = path.read_text(encoding="utf-8")
        self.tags, self.addresses, self.policy = set(), [], ""
        self.feed(self.source)

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        self.addresses += [value for name, value in attrs if name in ("href", "src")]
        if ("http-equiv", "Content-Security-Policy") in attrs:
            self.policy = dict(attrs)["content"]


def test_the_report_builds_from_a_small_scored_run_and_shows_the_standings(tmp_path):
    cfg, layout = scored_run(tmp_path)

    index = build_report(layout, cfg)

    assert index == layout.report_dir / "index.html"
    source = index.read_text(encoding="utf-8")
    assert standings(source) == STANDINGS
    shown = pictures(source)
    assert list(shown) == ORDER  # the pictures are best first too
    assert len({image.tobytes() for image in shown.values()}) == 3  # no two alike, so a picture under the wrong name would show
    for model, image in shown.items():
        typed = render_picture(load_picture(layout.attempt(SLUG, WIDTH, model, 1).picture_txt))
        assert (image.size, image.tobytes()) == (typed.size, typed.tobytes())  # as the judge was shown it
        assert attempt_page(layout, model).is_file()


def test_an_attempt_the_provider_blocked_is_shown_as_blocked_by_the_provider_and_not_as_the_model_declining(tmp_path):
    cfg, layout = scored_run(tmp_path, blocked={"refuse": "provider"}, decliner={"refuse": True})

    index = build_report(layout, cfg).read_text(encoding="utf-8")

    blocked, decliner = (attempt_page(layout, model).read_text(encoding="utf-8") for model in ("blocked", "decliner"))
    assert "Blocked by the provider: the platform stopped the request, the model did not decline." in blocked
    assert "Declined the task" not in blocked and "The refusal" not in blocked
    assert "Declined the task." in decliner and "Blocked by the provider" not in decliner
    # The index counts the two apart and gives each its own reason.
    assert "1 blocked by the provider, 1 refused" in index
    assert index.count("Blocked by the provider: the platform stopped the request, the model did not decline.") == 1
    assert index.count("Declined the task.") == 1


def test_no_page_holds_an_address_outside_itself_but_the_credit_of_the_reference_image(tmp_path):
    cfg, layout = scored_run(tmp_path)
    saved = layout.judgement_dir(SLUG, WIDTH, "judge") / "1.json"
    judgement = read_json(saved)
    lure = '<img src="https://elsewhere.example/x.png">'  # a judge's note is a model's text, and is printed on every page
    write_json(saved, {**judgement, "notes": dict.fromkeys(judgement["labels"], lure)})

    index = build_report(layout, cfg).resolve()

    pages = {path.resolve(): Page(path) for path in layout.report_dir.rglob("*.html")}
    assert len(pages) == 4
    reference = load_painting(SLUG, PAINTINGS).meta["reference"]
    assert reference["credit"] in pages[index].source
    for path, page in pages.items():
        assert "elsewhere.example" in page.source, path  # the note is there, as text
        links = {a.partition("#")[0] for a in page.addresses if not a.startswith(("#", "data:"))}
        elsewhere = {link for link in links if (path.parent / link).resolve() not in pages}  # not a page of the report
        assert elsewhere == ({reference["source_page_url"], reference["licence_url"]} if path == index else set()), path
        assert not page.tags & {"script", "iframe", "object"}, path
        assert "url(" not in page.source and "@import" not in page.source, path
        assert page.policy.startswith("default-src 'none'; img-src data:; style-src 'unsafe-inline'"), path  # the browser holds it to that


def test_a_run_with_a_missing_reference_image_still_builds_and_says_how_to_fetch_it(tmp_path):
    cfg, layout = scored_run(tmp_path)
    paintings = tmp_path / "paintings"
    (paintings / SLUG).mkdir(parents=True)
    shutil.copyfile(PAINTINGS / SLUG / "meta.yaml", paintings / SLUG / "meta.yaml")  # the painting is known; its image is not there

    source = build_report(layout, cfg, paintings_dir=paintings).read_text(encoding="utf-8")

    assert standings(source) == STANDINGS and list(pictures(source)) == ORDER
    assert "The real painting cannot be shown" in source
    assert f"python scripts/fetch_originals.py {SLUG}" in source
    assert all(attempt_page(layout, model).is_file() for model in ORDER)
