"""The human check: a blind sheet, the ranking a person saves from it, and the address it is served on."""

from __future__ import annotations

import base64
import http.client
import io
import json
import re
import threading

import pytest
import yaml
from PIL import Image

from paintbench import human
from paintbench.canvas import load_picture
from paintbench.importer import import_manifest
from paintbench.layout import read_json
from paintbench.paintings import load_painting
from paintbench.render import render_picture

from conftest import ROOT, picture_text

SLUG, WIDTH, PERSON = "great-wave", 32, "alice"
MODELS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]  # made-up contestants
CARD = re.compile(r'<li class="card"[^>]*data-label="([A-Z])">.*?<img src="data:image/png;base64,([A-Za-z0-9+/=]+)"', re.S)


@pytest.fixture
def layout(tmp_path):
    """Six typed pictures brought into a run by a manifest that also holds rankings of them, with a blind sheet for alice.

    Each picture.txt opens with a comment that names its model, as the harness writes it: that must not reach the sheet.
    """
    painting = load_painting(SLUG, ROOT / "paintings")
    for model in MODELS:
        typed = picture_text(WIDTH, painting.height_for(WIDTH), model)
        (tmp_path / f"{model}.txt").write_text(f"# model {model} ({model}-20261119), attempt 1\n{typed}", "utf-8")
    manifest = {
        "painting": SLUG, "width": WIDTH, "attempts": [{"model": model, "file": f"{model}.txt"} for model in MODELS],
        "rankings": [{"kind": "human", "name": "bob", "order": MODELS}, {"kind": "judge", "name": "judge-a", "order": MODELS[::-1]}],
    }
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest), "utf-8")
    layout = import_manifest(tmp_path / "manifest.yaml", tmp_path / "runs" / "typed", ROOT / "paintings")
    human.make_sheet(layout, painting, WIDTH, PERSON, seed=11)
    return layout


@pytest.fixture
def served(layout):
    """The port of the sheet's server, answering on the loopback address in the background."""
    server = human.make_server(layout, SLUG, WIDTH, PERSON, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()
    thread.join(5)


def ask(port: int, method: str, path: str, data: dict | None = None) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request(method, path, body=json.dumps(data) if data else None, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def record_of(layout) -> dict:
    """The sheet's record, which holds the key (letter -> attempt) and, once saved, the ranking."""
    return read_json(layout.human_path(SLUG, WIDTH, PERSON))


def names_a_model(text: str) -> bool:
    return any(model in text.lower() for model in MODELS)


def test_the_sheet_shows_each_picture_under_a_random_letter_and_names_no_model_in_its_source(layout):
    page, record = layout.human_sheet_path(SLUG, WIDTH, PERSON).read_text("utf-8"), record_of(layout)
    key, attempts = record["labels"], [attempt.id for attempt in layout.complete_attempts(SLUG, WIDTH)]
    cards = {label: base64.b64decode(data) for label, data in CARD.findall(page)}  # letter -> PNG, as the page shows them

    assert list(cards) == record["order_shown"] and sorted(key.values()) == sorted(attempts)  # every picture, once
    # Neither the letters nor the order shown follows the attempts. The fixture's sheet has a fixed seed, so this is certain.
    assert sorted(key) != list(human.LETTERS[:6]) and [key[label] for label in sorted(key)] != attempts
    assert [key[label] for label in cards] != attempts
    for label, png in cards.items():  # under each letter is the picture of the attempt the key gives it, and nothing else
        typed = load_picture(layout.attempt_from_id(key[label]).picture_txt)
        with Image.open(io.BytesIO(png)) as drawn:
            assert drawn.convert("RGB").tobytes() == render_picture(typed).tobytes()
            assert drawn.info == {}  # no text or other metadata came along in the image file
    assert not names_a_model(re.sub(r"base64,[A-Za-z0-9+/=]+", "", page))  # without the image bytes, which could spell a name by chance
    assert set(re.findall(r'\sdata-[a-z-]+="([^"]*)"', page)) == set(key)  # the only data the page carries is the letters


def test_a_ranking_saved_from_the_served_page_is_recorded_against_the_attempts_behind_its_letters(layout, served):
    key = record_of(layout)["labels"]
    ranking = sorted(key, reverse=True)
    assert ask(served, "GET", "/") == (200, layout.human_sheet_path(SLUG, WIDTH, PERSON).read_bytes())  # the blind sheet as it is

    status, reply = ask(served, "POST", "/rank", {"ranking": ranking[1:]})  # a letter left out
    assert status == 400 and not names_a_model(reply.decode()) and record_of(layout)["ranking"] is None  # nothing stored
    status, reply = ask(served, "POST", "/rank", {"ranking": ranking, "notes": "the top two were close"})
    assert (status, json.loads(reply)) == (200, {"ok": True})  # never the record, which holds the key
    record = record_of(layout)
    assert record["ranking"] == ranking and record["ranking_attempts"] == [key[label] for label in ranking]
    assert record["notes"] == "the top two were close" and record["labels"] == key


def test_the_server_listens_on_the_one_address_it_is_given_and_refuses_every_interface(layout):
    server = human.make_server(layout, SLUG, WIDTH, PERSON, "127.0.0.1", 0)
    server.server_close()
    assert server.server_address[0] == "127.0.0.1"
    for everywhere in ("0.0.0.0", "::", "", "0", "::ffff:0.0.0.0"):
        with pytest.raises(ValueError):  # the same call, so only the address can be what is refused
            human.make_server(layout, SLUG, WIDTH, PERSON, everywhere, 0).server_close()
