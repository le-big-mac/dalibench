"""Fetch the museum photographs the public repository leaves out, and rebuild missing reference images.

    python scripts/fetch_originals.py [slug ...]        every painting when none is named

For each painting whose original is absent, the file at `reference.download_url` in its meta.yaml is
downloaded, checked against `reference.original_sha256`, and saved as `reference.original_file`. Then,
if the working copy the judges see (reference.png) is absent, it is built from the original and compared
with `reference.reference_sha256`. One line is printed per painting. The exit status is 1 if any failed.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # run as a script, the package is found beside this directory
    sys.path.insert(0, str(REPO))

from paintbench.paintings import default_paintings_dir, load_painting  # noqa: E402
from paintbench.reference import prepare_reference, sha256_file  # noqa: E402

USER_AGENT = "dalibench-fetch-originals/1.0 (+https://github.com/le-big-mac/dalibench)"  # some hosts turn away a client that does not say what it is
TIMEOUT_SECONDS = 60  # for the connection, and for each read from it
DEADLINE_SECONDS = 900  # for one whole download
MAX_BYTES = 64 * 1024 * 1024  # the largest original is under 18 MB


class Failed(Exception):
    """What went wrong with one painting, in words for the person running the script."""


def download(url: str, part: Path) -> tuple[str, int, bool]:
    """Save what `url` serves at `part`. Returns its SHA-256, its size, and whether it looks like a web page, not an image."""
    if not url.startswith(("https://", "http://")):
        raise Failed(f"{url!r} is not a web address")
    digest, size, head, started = hashlib.sha256(), 0, b"", time.monotonic()
    try:
        with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=TIMEOUT_SECONDS) as response, open(part, "wb") as out:
            kind = str(response.headers.get("Content-Type") or "").lower()
            while chunk := response.read(1 << 16):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise Failed(f"{url} serves more than {MAX_BYTES:,} bytes, the most this script takes")
                if time.monotonic() - started > DEADLINE_SECONDS:
                    raise Failed(f"{url} was still arriving after {DEADLINE_SECONDS} seconds")
                head = head or chunk[:16]
                digest.update(chunk)
                out.write(chunk)
    except HTTPError as e:
        raise Failed(f"{url} could not be downloaded (HTTP {e.code} {e.reason})") from e
    except URLError as e:
        raise Failed(f"{url} could not be downloaded ({e.reason})") from e
    except (OSError, http.client.HTTPException) as e:
        raise Failed(f"{url} could not be downloaded ({str(e) or type(e).__name__})") from e
    return digest.hexdigest(), size, kind.startswith("text/") or head.lstrip().startswith(b"<")


def fetch_original(slug: str, url: str, recorded: str, original: Path, working: Path) -> str:
    """Download the original and save it if it is the recorded file. Returns what was done; Failed says what to do by hand."""
    spare = f"{working.name} is present, and only rebuilding it needs the original"
    by_hand = f"By hand: open {url} in a browser, save the image as {original}, and run this script again to check the file"
    by_hand += f". {spare}" if working.is_file() else f" and build {working.name}"
    part = original.with_name(original.name + ".part")
    try:
        try:
            sha256, size, page = download(url, part)
        except Failed as e:
            raise Failed(f"{e}. {by_hand}") from e
        if sha256 != recorded and page:
            raise Failed(f"{url} answered with a web page ({size:,} bytes), not the image: the site may turn automated clients away. {by_hand}")
        if sha256 != recorded:
            other = f"{url} serves a file that is not the one meta.yaml records (SHA-256 {sha256}, recorded {recorded}), so it was not saved"
            if working.is_file():
                raise Failed(f"{other}. {spare}. To keep the file all the same, save that address as {original} by hand")
            raise Failed(
                f"{other}. To use it all the same: save that address as {original} by hand, then `paintbench reference {slug}` "
                f"builds {working.name} from it, which will not be the published working copy"
            )
        part.replace(original)
    finally:
        part.unlink(missing_ok=True)
    return f"downloaded {original.name} ({size:,} bytes, SHA-256 as recorded)"


def _plain_name(name: str) -> bool:
    """True for one file or directory name: nothing that could lead out of the directory it is joined to."""
    return bool(name) and name not in (".", "..") and Path(name).name == name


def fetch_painting(slug: str, paintings_dir: Path) -> str:
    """Fetch and build what one painting lacks. Returns the line to print; Failed if it could not be done."""
    if not _plain_name(slug):
        raise Failed(f"{slug!r} is not the name of a painting's directory")
    try:
        painting = load_painting(slug, paintings_dir)
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise Failed(str(e)) from e
    ref = painting.meta.get("reference") or {}
    for key in ("original_file", "original_sha256"):
        if not ref.get(key):
            raise Failed(f"{painting.directory / 'meta.yaml'}: reference.{key} is not set")
    if not _plain_name(str(ref["original_file"])):  # files are written inside the painting's own directory and nowhere else
        raise Failed(f"{painting.directory / 'meta.yaml'}: reference.original_file must be a file name, not a path")
    original, working = painting.directory / str(ref["original_file"]), painting.reference_path
    url, recorded = str(ref.get("download_url") or ""), str(ref["original_sha256"])

    done = []
    if not original.is_file():
        if not url:
            raise Failed(f"{painting.directory / 'meta.yaml'}: reference.download_url is not set, and there is no {original.name}")
        done.append(fetch_original(slug, url, recorded, original, working))
    else:
        sha256 = sha256_file(original)
        if sha256 != recorded:
            build = "" if working.is_file() else f", or build {working.name} from it all the same with `paintbench reference {slug}`"
            where = f" with the file at {url}" if url else ""
            raise Failed(f"{original} is not the file meta.yaml records (SHA-256 {sha256}, recorded {recorded}). Replace it{where}{build}")

    if not working.is_file():
        try:
            built = prepare_reference(original, working, ref.get("crop_box"))["reference_sha256"]
        except (OSError, ValueError) as e:
            raise Failed(f"{working.name} could not be built from {original.name}: {e}") from e
        published = ref.get("reference_sha256")
        if built == published:
            done.append(f"built {working.name} (SHA-256 as recorded)")
        elif not published:
            done.append(f"built {working.name} (meta.yaml records no SHA-256 to compare it with)")
        else:
            done.append(
                f"built {working.name}. WARNING: this working copy differs from the published one (SHA-256 {built}, published {published}). "
                "Image libraries differ between versions, so the same original can build to a slightly different file. The file is kept."
            )
    return f"{slug}: {'; '.join(done)}" if done else f"{slug}: nothing to do ({original.name} is the recorded file, and {working.name} is present)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the originals the public repository leaves out, and rebuild missing reference images.")
    parser.add_argument("slugs", nargs="*", metavar="slug", help="a painting's directory name under paintings/ (default: every painting)")
    parser.add_argument("--paintings-dir", help="where the paintings are (default: $PAINTBENCH_PAINTINGS, else ./paintings, else beside the package)")
    args = parser.parse_args(argv)
    paintings_dir = Path(args.paintings_dir) if args.paintings_dir else default_paintings_dir()
    slugs = args.slugs or sorted(path.parent.name for path in paintings_dir.glob("*/meta.yaml"))
    if not slugs:
        parser.error(f"there are no paintings in {paintings_dir}")
    failed = False
    for slug in slugs:
        try:
            line = fetch_painting(slug, paintings_dir)
        except Failed as e:
            failed, line = True, f"{slug}: FAILED: {e}."
        print(line, file=sys.stderr if "FAILED: " in line or "WARNING: " in line else sys.stdout, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
