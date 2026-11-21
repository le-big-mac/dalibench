"""The sixteen paintings: what a model is told of each, the grid it types on, the reference image, and the copyright status."""

from __future__ import annotations

import re
from datetime import date

from paintbench.paintings import STANDARD_WIDTHS, load_painting
from paintbench.reference import sha256_file

from conftest import ROOT, left_out_of_export

PAINTINGS = ROOT / "paintings"
SLUGS = sorted(path.parent.name for path in PAINTINGS.glob("*/meta.yaml"))
IN_COPYRIGHT = ["persistence-of-memory"]
# The text after "Painting: " in the task message: all a model is told of the work.
FIRST_LINES = {
    "birth-of-venus": "The Birth of Venus, by Sandro Botticelli, about 1485.",
    "fighting-temeraire": "The Fighting Temeraire, by J. M. W. Turner, 1839.",
    "girl-with-a-pearl-earring": "Girl with a Pearl Earring, by Johannes Vermeer, about 1665.",
    "grande-jatte": "A Sunday Afternoon on the Island of La Grande Jatte, by Georges Seurat, 1884 to 1886.",
    "great-wave": "Under the Wave off Kanagawa (often called The Great Wave), by Katsushika Hokusai, about 1831.",
    "mona-lisa": "Mona Lisa, by Leonardo da Vinci, about 1503 to 1519.",
    "night-watch": "The Night Watch, by Rembrandt van Rijn, 1642.",
    "persistence-of-memory": "The Persistence of Memory, by Salvador Dalí, 1931.",
    "red-fuji": "Fine Wind, Clear Morning (often called Red Fuji), by Katsushika Hokusai, about 1831.",
    "starry-night": "The Starry Night, by Vincent van Gogh, 1889.",
    "sudden-shower": "Sudden Shower over Shin-Ōhashi Bridge and Atake, by Utagawa Hiroshige, 1857.",
    "sunflowers-1888": "Sunflowers, by Vincent van Gogh, 1888. This is the 1888 version in the National Gallery, London.",
    "the-kiss": "The Kiss, by Gustav Klimt, 1907 to 1908.",
    "the-scream": "The Scream, by Edvard Munch, 1893. This is the 1893 painted version in the National Museum, Oslo.",
    "wanderer-above-the-sea-of-fog": "Wanderer above the Sea of Fog, by Caspar David Friedrich, about 1818.",
    "water-lilies-1906": "Water Lilies, by Claude Monet, 1906. This is the 1906 painting in the Art Institute of Chicago, one of a series of about 250.",
}


def test_what_a_model_is_told_of_each_painting_is_fixed():
    assert {slug: load_painting(slug, PAINTINGS).first_line for slug in SLUGS} == FIRST_LINES


def test_every_painting_has_its_metadata_and_grid_heights_that_follow_from_its_reference_shape():
    for slug in SLUGS:
        painting = load_painting(slug, PAINTINGS)
        reference = painting.meta["reference"]
        for key in ("source_page_url", "download_url", "original_sha256", "reference_sha256", "credit"):
            assert reference.get(key), f"{slug}: reference.{key} is missing"
        wide, high = reference["cropped_pixel_size"]
        assert set(STANDARD_WIDTHS) <= set(painting.grid_heights), slug
        for width, height in painting.grid_heights.items():  # the reference's shape at that width, a half rounding up
            assert height == (2 * width * high + wide) // (2 * wide), f"{slug} at {width} wide"


def test_each_reference_image_present_is_the_file_its_record_names():
    checked = 0
    for slug in SLUGS:
        painting = load_painting(slug, PAINTINGS)
        if not painting.reference_path.is_file() and left_out_of_export(painting.reference_path):
            continue  # the public repository leaves this one out
        assert sha256_file(painting.reference_path) == painting.meta["reference"]["reference_sha256"], slug
        checked += 1
    assert checked >= len(SLUGS) - len(IN_COPYRIGHT)


def test_a_painting_in_copyright_says_so_and_every_other_is_in_the_public_domain():
    for slug in SLUGS:
        painting = load_painting(slug, PAINTINGS)
        status, reference = painting.meta["public_domain"], painting.meta["reference"]
        if slug in IN_COPYRIGHT:
            assert status["in_public_domain"] is False
            assert not status.get("artist_died")  # a year under this key would mark the work as in the public domain
            assert reference["licence_caveat"].startswith("LICENCE DOUBT")  # the words that mark an image whose rights are in doubt
            assert left_out_of_export(painting.reference_path)  # and the public repository does not carry the image
        else:  # the rule: made before 1929, by an artist who died more than 70 years ago
            assert int(re.findall(r"\d{4}", str(status["made"]))[-1]) < 1929, slug
            assert date.today().year - int(status["artist_died"]) > 70, slug
