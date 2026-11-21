"""Build the working copy of a reference image from the downloaded original.

The working copy is the original cropped to the painted surface, converted to
sRGB, with the longest side at 1,024 pixels, saved as PNG. One reference per
work; it is never swapped mid-study.

Rules:

- The same original and crop give the same file every time. The working copy
  carries no colour profile: its pixels are already sRGB, and the sRGB profile
  the conversion would attach holds the second it was made, which would change
  the file hash on every build.
- A working copy that already holds exactly the pixels a build produces is
  left as it is, byte for byte, so rebuilding never changes a reference in
  use. (The same pixels can be stored as different bytes: an older file with
  an embedded profile, another version of the compression library.)
- An original is used only if its colours can be converted reliably: 8 bits a
  channel, nothing hidden behind transparency, and an embedded profile that
  can be applied. Anything else is refused with what to do about it; a wrong
  or blank reference is never written.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageCms, ImageOps

WORKING_LONGEST_SIDE = 1024

# Modes Pillow holds at 8 bits a channel and converts to RGB exactly. (16-bit RGB files open as "RGB"
# with the high byte kept; 16-bit greyscale opens as "I;16", which a plain convert clips to white.)
_MODES = ("RGB", "RGBA", "L", "LA", "P", "CMYK")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_srgb(img: Image.Image, name: str) -> Image.Image:
    """An 8-bit RGB image in sRGB, with no profile attached; ValueError if that cannot be done reliably."""
    if img.mode not in _MODES:
        raise ValueError(
            f"{name}: image mode {img.mode!r} (16-bit greyscale, floating point, Lab and the like) cannot be "
            "converted to sRGB reliably. Convert the original to 8-bit RGB first."
        )
    if img.mode == "P" or "transparency" in img.info:
        img = img.convert("RGBA")  # brings palette and colour-key transparency into an alpha channel
    if img.mode in ("RGBA", "LA"):
        if img.getchannel("A").getextrema()[0] < 255:
            raise ValueError(f"{name}: the image has transparent pixels, and what lies under them is not part of the picture. Flatten the original first.")
        img = img.convert(img.mode[:-1])
    icc = img.info.get("icc_profile")
    if icc:
        try:
            img = ImageCms.profileToProfile(img, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"), outputMode="RGB")
        except (ImageCms.PyCMSError, OSError, ValueError) as e:
            raise ValueError(f"{name}: the embedded colour profile cannot be applied ({e}). Convert the original to sRGB first.") from e
    elif img.mode == "CMYK":
        raise ValueError(f"{name}: a CMYK image with no colour profile has no defined colours. Convert the original to sRGB first.")
    img = img.convert("RGB")
    img.info.pop("icc_profile", None)
    return img


def _holds(path: Path, img: Image.Image) -> bool:
    """True if the PNG at `path` already holds exactly these pixels."""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as old:
            return old.format == "PNG" and old.mode == img.mode and old.size == img.size and old.tobytes() == img.tobytes()
    except OSError:
        return False


def prepare_reference(
    original_path: Path | str,
    out_path: Path | str,
    crop_box: list[int] | tuple[int, int, int, int] | None = None,
    longest_side: int = WORKING_LONGEST_SIDE,
) -> dict[str, Any]:
    """Write the working copy and return the facts to record in meta.yaml."""
    original_path, out_path = Path(original_path), Path(out_path)
    Image.MAX_IMAGE_PIXELS = 400_000_000  # museum originals are large, and each is a file that meta.yaml names
    with Image.open(original_path) as img:
        img = ImageOps.exif_transpose(img)
        original_size = [img.width, img.height]
        had_profile = bool(img.info.get("icc_profile"))
        img = _to_srgb(img, original_path.name)
        if crop_box:
            left, top, right, bottom = (int(v) for v in crop_box)
            if not (0 <= left < right <= img.width and 0 <= top < bottom <= img.height):
                raise ValueError(f"crop box {list(crop_box)} does not fit the {img.width} x {img.height} original")
            img = img.crop((left, top, right, bottom))
        cropped_size = [img.width, img.height]
        scale = longest_side / max(img.width, img.height)
        if scale < 1:
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
        if not _holds(out_path, img):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="PNG")
        working_size = [img.width, img.height]
    return {
        "original_sha256": sha256_file(original_path),
        "original_pixel_size": original_size,
        "original_had_colour_profile": had_profile,
        "crop_box": list(crop_box) if crop_box else [0, 0, original_size[0], original_size[1]],
        "cropped_pixel_size": cropped_size,
        "reference_sha256": sha256_file(out_path),
        "reference_pixel_size": working_size,
        "height_over_width": cropped_size[1] / cropped_size[0],
    }
