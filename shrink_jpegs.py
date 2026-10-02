#!/usr/bin/env python3
"""Shrink raster images and pack them into a single zip under a target size.

Supports JPEG, PNG, BMP, GIF, TIFF, WEBP, and HEIC/HEIF input. Output is
always JPEG -- vector formats (SVG, etc.) are out of scope.

Given a target zip size and a folder of images, this computes how small
each image needs to be -- not a flat per-file target, and not simply
proportional to current file size, but a fair-share ("water-filling")
allocation weighted by estimated content complexity. Images that are
already small enough for their allocated share keep their original size
untouched, and the headroom they don't use gets redistributed to the
images that need to shrink more.

Complexity weighting: on-disk file size conflates how visually complex
an image is with whatever quality/resolution/format it happened to be
saved at. To correct for this, each image gets a cheap complexity
probe: encode a downscaled thumbnail at a fixed quality, then scale the
byte count up by (original_pixels / thumbnail_pixels) to estimate
"bytes this image would need at a fixed quality, at full resolution."
Formats with draft-mode scaled decoding (JPEG, and HEIC via pillow-heif)
get the cheap path for this -- Image.draft() decodes directly at a
reduced scale, skipping a full decode. Every other format (PNG, BMP,
GIF, TIFF, WEBP) gets a full decode -- slower per file, but correct.
Either way the result is resized to the same thumbnail bound, so
estimates are comparable across formats.

Transparency: images with an alpha channel or a transparent colour key
are composited onto a white background before conversion to RGB. A
plain mode conversion to RGB would just discard the transparency and
keep whatever RGB values sit underneath it, which for many PNGs/GIFs
are garbage or black -- that produces a visibly wrong result rather
than the sane white-background flattening most people expect from a
JPEG conversion.

Orientation and colour: re-encoded JPEGs carry no EXIF, so the EXIF
orientation is baked into the pixels first -- otherwise phone photos
come out sideways. An RGB ICC profile (e.g. Display P3 on iPhone
photos) is carried over so colours don't shift; gray/CMYK profiles no
longer describe the converted RGB pixels and are dropped. 16-bit
grayscale is scaled to 8-bit rather than clipped.

Animation: GIF, animated WEBP, and multi-page TIFF are flattened to
their first frame -- JPEG has no concept of animation. This is logged
per file so it isn't a silent surprise.

Each image is shrunk (quality first, then dimensions) in parallel
worker processes, and the results are packed into shrunk_images.zip
using ZIP_STORED -- the source data is either already-compressed
(JPEG/HEIC) or about to be freshly JPEG-encoded, so DEFLATE would
spend CPU for no benefit either way. The size budget reserves an upper
bound for the zip's own per-entry metadata.

Non-destructive: originals are never modified. Shrunk copies are
produced in a temp directory and packed into the zip; the temp
directory is cleaned up automatically. The zip itself is written to a
temp name and renamed into place, so a failed run never leaves a
truncated zip behind.

Dependency note: HEIC decoding is not built into Pillow. This script
requires the `pillow-heif` package (`pip install pillow-heif`) only for
.heic/.heif input -- the other formats are handled by Pillow core. If
you're targeting an air-gapped host, vendor that wheel ahead of time --
this script does not fetch it for you.
"""

from __future__ import annotations

import argparse
import io
import itertools
import logging
import math
import os
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

logger = logging.getLogger("shrink_jpegs")

MIN_QUALITY = 20
MAX_QUALITY = 95
QUALITY_STEP = 5
DIM_SCALE_FACTOR = 0.9
MAX_DIM_ITERATIONS = 10
MIN_DIM = 50

JPEG_SUFFIXES = (".jpg", ".jpeg")
HEIF_SUFFIXES = (".heic", ".heif")
OTHER_RASTER_SUFFIXES = (".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
ALL_SUFFIXES = JPEG_SUFFIXES + HEIF_SUFFIXES + OTHER_RASTER_SUFFIXES

# Pillow format names whose bytes are already a valid JPEG. MPO is a JPEG
# with extra embedded images appended (e.g. iPhone portrait depth maps).
JPEG_FORMATS = ("JPEG", "MPO")

PROBE_MAX_DIM = 400
PROBE_QUALITY = 75

# Upper bounds on zip bytes outside the file data. Per entry: local header
# (30) + central directory header (46) + worst-case zip64 extra fields
# (20 + 28) + data descriptor (24), plus the name stored twice. Per
# archive: end-of-central-directory record (22) + zip64 end record and
# locator (56 + 20).
ZIP_ENTRY_OVERHEAD = 30 + 46 + 20 + 28 + 24
ZIP_END_OVERHEAD = 22 + 56 + 20


def is_native_jpeg(path: Path) -> bool:
    """True for .jpg/.jpeg file names.

    Used for naming: these keep their name in the zip, everything else is
    renamed to .jpg. Whether a file can be copied unchanged is decided
    from its decoded format instead, so a mislabeled PNG still gets
    re-encoded.
    """
    return path.suffix.lower() in JPEG_SUFFIXES


def zip_overhead_bytes(arcnames: Iterable[str]) -> int:
    """Upper bound on a ZIP_STORED archive's size beyond its file data."""
    return ZIP_END_OVERHEAD + sum(
        ZIP_ENTRY_OVERHEAD + 2 * len(name.encode("utf-8")) for name in arcnames
    )


def rgb_icc_profile(img: Image.Image) -> bytes | None:
    """Return img's embedded ICC profile if it describes an RGB colour space.

    Output is always RGB, so an RGB source profile (e.g. Display P3 on
    iPhone photos) still describes the pixels and has to be carried over
    or colours shift. Gray/CMYK profiles don't match the converted pixels.
    """
    icc = img.info.get("icc_profile")
    # Bytes 16-19 of an ICC header are the data colour space signature.
    if isinstance(icc, bytes) and icc[16:20] == b"RGB ":
        return icc
    return None


def load_as_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow image mode to RGB, compositing transparency onto white.

    img.convert("RGB") on a transparent source just discards the alpha
    channel and keeps whatever RGB values sit underneath it -- for many
    PNGs/GIFs those pixels are garbage colors (often black), producing a
    visibly wrong result. Compositing onto white first gives a sane,
    predictable flattening instead. Transparency can come from an alpha
    band (RGBA/LA/PA) or a colour key in img.info (palette images and
    tRNS-keyed L/RGB PNGs).

    16-bit grayscale is scaled to 8-bit first -- Pillow's own conversion
    clips everything above 255, washing the image out to white.
    """
    if img.mode == "I" or img.mode.startswith("I;16"):
        img = img.convert("I").point(lambda v: v * (1 / 256)).convert("L")

    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    if not has_alpha:
        return img.convert("RGB")

    img = img.convert("RGBA")
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[-1])
    return background


def estimate_complexity_weight(path: Path) -> tuple[int, str | None]:
    """Estimate a fair-share weight for one image based on encode complexity.

    For JPEG (and HEIC via pillow-heif) input, Image.draft() decodes
    directly at a reduced scale instead of a full decode -- this is what
    keeps the probe cheap relative to the real shrink pass. Other formats
    ignore draft() and get a full decode -- slower per file, but the
    estimate is still correct. Every format is then thumbnailed to the
    same bound, since draft only scales in coarse steps (1/2-1/8) and
    bytes-per-pixel varies with resolution. The thumbnail is encoded once
    at a fixed quality (not searched), and the resulting byte count is
    scaled up by (original_pixels / thumbnail_pixels) to estimate "bytes
    this image would need at a fixed quality, at full resolution."

    Returns (weight_bytes, error_or_None). Callers should fall back to
    on-disk file size as the weight if error is not None.
    """
    try:
        with Image.open(path) as img:
            original_w, original_h = img.size  # header read only, no decode yet
            original_pixels = original_w * original_h
            if original_pixels == 0:
                raise ValueError("zero-pixel image")

            img.draft("RGB", (PROBE_MAX_DIM, PROBE_MAX_DIM))
            thumb = load_as_rgb(img)
        thumb.thumbnail((PROBE_MAX_DIM, PROBE_MAX_DIM), Image.LANCZOS)

        thumb_pixels = thumb.size[0] * thumb.size[1]
        if thumb_pixels == 0:
            raise ValueError("thumbnail collapsed to zero pixels")

        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=PROBE_QUALITY)
        thumb_bytes = buf.tell()

        weight = int(thumb_bytes * (original_pixels / thumb_pixels))
        return weight, None
    except Exception as exc:
        return 0, f"complexity probe failed for '{path}': {exc}"


def compute_fair_share_targets(
    file_sizes: dict[Path, int],
    weights: dict[Path, int],
    total_budget_bytes: int,
) -> dict[Path, int]:
    """Water-filling fair-share allocation, weighted by complexity.

    Each remaining file's proportional share of the budget is
    weight-proportional rather than equal. Files whose proportional
    share already covers their actual current size are "settled": they
    keep their own size untouched, and that size is subtracted from the
    budget (and their weight removed from the pool) so the remaining
    files' shares are recalculated against what's left. Repeats until
    no more files settle; whatever remains is split by weight.

    Guarantees sum(targets) <= total_budget_bytes (assuming at least
    one file and total_budget_bytes >= 0).
    """
    if not file_sizes:
        return {}

    remaining_sizes = dict(file_sizes)
    remaining_weights = dict(weights)
    budget = total_budget_bytes
    targets: dict[Path, int] = {}

    while remaining_sizes:
        weight_sum = sum(remaining_weights.values())
        if weight_sum <= 0:
            # No usable weight info left (e.g. every remaining probe
            # failed) -- fall back to an even split of what's left.
            avg = max(0, budget) // len(remaining_sizes)
            for p in remaining_sizes:
                targets[p] = avg
            break

        proportional = {
            p: (max(0, budget) * remaining_weights[p]) // weight_sum
            for p in remaining_sizes
        }

        settled = {p: s for p, s in remaining_sizes.items() if proportional[p] >= s}
        if not settled:
            for p in remaining_sizes:
                targets[p] = proportional[p]
            break

        for p, s in settled.items():
            targets[p] = s
            budget -= s
            del remaining_sizes[p]
            del remaining_weights[p]

    return targets


def _encode_jpeg(img: Image.Image, quality: int, icc_profile: bytes | None) -> bytes:
    buf = io.BytesIO()
    img.save(
        buf, format="JPEG", quality=quality, optimize=True, icc_profile=icc_profile,
    )
    return buf.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    """Write via a sibling temp file + rename, so path is never left truncated."""
    tmp_path = path.with_name(path.name + ".tmp_shrink")
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def compress_to_target(
    src_path: Path, final_dst_path: Path, target_bytes: int
) -> tuple[bool, int]:
    """Compress/convert a single image to a JPEG under target_bytes.

    JPEG data that already fits is copied byte-for-byte. Everything else
    is re-encoded: quality first, then dimensions, each downscale jumping
    toward the size the byte overshoot implies.

    Always writes to a temp file next to final_dst_path first, then
    renames into place on success -- this is safe even when
    final_dst_path == src_path, since the source is fully decoded before
    anything is written.

    Returns (success, final_size_bytes). final_dst_path is only
    written if a passing result is found.
    """
    if target_bytes < 0:
        raise ValueError(f"target_bytes must be non-negative: {target_bytes}")

    final_dst_path.parent.mkdir(parents=True, exist_ok=True)

    img: Image.Image | None = None  # stays None on the copy-unchanged path
    icc_profile: bytes | None = None
    try:
        with Image.open(src_path) as src_img:
            # Decided from the decoded format, not the suffix -- a PNG named
            # .jpg must still be re-encoded, since the output must be JPEG.
            is_jpeg = src_img.format in JPEG_FORMATS
            if not (is_jpeg and src_path.stat().st_size <= target_bytes):
                n_frames = getattr(src_img, "n_frames", 1)
                if n_frames > 1 and not is_jpeg:
                    logger.warning(
                        "'%s' has %d frames -- using the first frame only, "
                        "JPEG output can't preserve animation",
                        src_path, n_frames,
                    )
                icc_profile = rgb_icc_profile(src_img)
                img = load_as_rgb(ImageOps.exif_transpose(src_img))
    except Exception as exc:
        raise OSError(f"failed to open '{src_path}': {exc}") from exc

    if img is None:
        data = src_path.read_bytes()
        logger.info(
            "'%s' already %d bytes (<= target %d), copying unchanged",
            src_path, len(data), target_bytes,
        )
        if final_dst_path != src_path:
            _write_atomic(final_dst_path, data)
        return True, len(data)

    current_img = img
    smallest = 0
    for _ in range(MAX_DIM_ITERATIONS + 1):
        # Size falls with quality, so one encode at the floor shows whether
        # any quality can fit at these dimensions before searching down.
        floor_data = _encode_jpeg(current_img, MIN_QUALITY, icc_profile)
        if len(floor_data) <= target_bytes:
            data, quality = floor_data, MIN_QUALITY
            for q in range(MAX_QUALITY, MIN_QUALITY, -QUALITY_STEP):
                candidate = _encode_jpeg(current_img, q, icc_profile)
                if len(candidate) <= target_bytes:
                    data, quality = candidate, q
                    break
            _write_atomic(final_dst_path, data)
            logger.info(
                "'%s' -> %d bytes at quality=%d, dims=%s",
                src_path, len(data), quality, current_img.size,
            )
            return True, len(data)
        smallest = len(floor_data)

        # Quality alone didn't get there; downscale dimensions and retry.
        # Size scales roughly with pixel count, so jump to the linear scale
        # the overshoot implies (with DIM_SCALE_FACTOR as margin) rather than
        # fixed steps, which run out of iterations on heavy reductions. Never
        # go below MIN_DIM, and resample from the full-size image so blur
        # doesn't compound across steps.
        w, h = current_img.size
        step = DIM_SCALE_FACTOR * math.sqrt(target_bytes / len(floor_data))
        step = max(step, MIN_DIM / min(w, h))
        if step >= 1:
            break
        new_w, new_h = max(1, round(w * step)), max(1, round(h * step))
        current_img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(
            "'%s' still over target after quality pass, downscaling to %dx%d",
            src_path, new_w, new_h,
        )

    # Failed to hit target even at min quality + min dimensions.
    logger.error(
        "'%s' could not be shrunk below %d bytes (best: %d bytes at min quality/dims)",
        src_path, target_bytes, smallest,
    )
    return False, smallest


def _worker_init(log_level: int) -> None:
    """Configure logging inside each worker process.

    Needed because worker processes don't inherit the parent's already-
    configured logging on platforms that use 'spawn' (macOS, Windows) --
    each worker starts fresh and needs its own basicConfig call.
    """
    logging.basicConfig(
        level=log_level, format="%(asctime)s [%(levelname)s] %(message)s"
    )


def _probe_one(path: Path) -> tuple[Path, int, str | None]:
    """Picklable wrapper around estimate_complexity_weight for a worker process."""
    weight, err = estimate_complexity_weight(path)
    return path, weight, err


def _process_one(
    src_path: Path, dst_path: Path, target_bytes: int
) -> tuple[bool, int, str | None]:
    """Picklable wrapper around compress_to_target for use in a worker process."""
    try:
        ok, size = compress_to_target(src_path, dst_path, target_bytes)
        return ok, size, None
    except OSError as exc:
        return False, 0, str(exc)
    except Exception as exc:
        # Anything else (decoder bug, MemoryError) is reported against its
        # own file instead of aborting the whole run.
        return False, 0, f"failed to process '{src_path}': {exc!r}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Shrink raster images (jpg/jpeg/png/bmp/gif/tif/tiff/webp/heic/heif) "
            "and pack them into a single zip under a target size, "
            "auto-calculating per-image targets weighted by content complexity."
        ),
    )
    parser.add_argument(
        "input_dir", type=Path,
        help="Directory containing raster image files",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("shrunk_images.zip"),
        help="Path to the output zip file (default: ./shrunk_images.zip)",
    )
    parser.add_argument(
        "--target-zip-mb", type=float, default=50,
        help="Target max size of the final zip in MB (default: 50)",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Recurse into subdirectories",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Max images to process concurrently (default: CPU core count). "
            "This is CPU-bound work and each worker holds a fully decoded "
            "image in memory -- workers beyond core count just use extra "
            "memory without going faster."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging",
    )
    args = parser.parse_args()

    if not math.isfinite(args.target_zip_mb) or args.target_zip_mb <= 0:
        parser.error("--target-zip-mb must be a finite number greater than 0")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.output.exists() and args.output.is_dir():
        logger.error("output path '%s' is a directory, not a file", args.output)
        return 1

    if not args.input_dir.is_dir():
        logger.error("input_dir '%s' is not a directory", args.input_dir)
        return 1

    pattern = "**/*" if args.recursive else "*"
    # Sorted so zip names, collision resolution, and logs are deterministic.
    files = sorted(
        p for p in args.input_dir.glob(pattern)
        if p.suffix.lower() in ALL_SUFFIXES and p.is_file()
    )

    heif_files_present = any(p.suffix.lower() in HEIF_SUFFIXES for p in files)
    if heif_files_present and not HEIF_SUPPORT:
        logger.error(
            "HEIC/HEIF files found but 'pillow-heif' is not installed. "
            "Install it with: pip install pillow-heif"
        )
        return 1

    if not files:
        logger.warning(
            "no supported raster image files found in '%s'", args.input_dir
        )
        return 0

    # The zip is renamed into place at the end -- never onto a source image.
    output_resolved = args.output.resolve()
    if any(p.resolve() == output_resolved for p in files):
        logger.error("output path '%s' is one of the input images", args.output)
        return 1

    cpu_count = os.cpu_count() or 1
    requested_workers = cpu_count if args.workers is None else args.workers
    workers = min(requested_workers, len(files))
    logger.info(
        "%d worker(s) requested (%d CPU cores detected)", workers, cpu_count,
    )
    if workers > cpu_count:
        logger.warning(
            "worker count (%d) exceeds detected CPU cores (%d) -- this is "
            "CPU-bound work, so throughput won't scale past core count",
            workers, cpu_count,
        )

    file_sizes = {p: p.stat().st_size for p in files}
    current_total = sum(file_sizes.values())

    # Zip names. Multiple source formats can share a stem (e.g. photo.bmp
    # and photo.webp both naively becoming photo.jpg), so only
    # disambiguate on an actual collision -- keeps the common single-
    # format case's output names clean. Native JPEGs are named first so
    # they keep their real names, and collisions are checked
    # case-insensitively since the zip will often be extracted onto a
    # case-insensitive filesystem (photo.JPG vs photo.png -> photo.jpg).
    arcnames: dict[Path, str] = {}
    used_keys: set[str] = set()
    for src in sorted(files, key=lambda p: not is_native_jpeg(p)):
        rel = src.relative_to(args.input_dir)
        natural = rel if is_native_jpeg(src) else rel.with_suffix(".jpg")
        candidates = itertools.chain(
            [natural, natural.parent / f"{rel.name}.jpg"],
            (natural.parent / f"{rel.name}_{i}.jpg" for i in itertools.count(2)),
        )
        arcname = next(
            c.as_posix() for c in candidates
            if c.as_posix().casefold() not in used_keys
        )
        if arcname != natural.as_posix():
            logger.warning(
                "'%s' -- name collision in the zip against another file's "
                "'%s', writing as '%s' instead",
                src, natural.as_posix(), arcname,
            )
        used_keys.add(arcname.casefold())
        arcnames[src] = arcname

    target_zip_bytes = int(args.target_zip_mb * 1_000_000)
    # Reserve headroom for zip metadata (local headers, central directory)
    # -- small per file, but scales with file count and name length, so
    # never less than the computed upper bound.
    safety_margin = max(
        50_000, int(target_zip_bytes * 0.01), zip_overhead_bytes(arcnames.values()),
    )
    total_budget = target_zip_bytes - safety_margin
    if total_budget <= 0:
        logger.error(
            "--target-zip-mb %g is too small: it must exceed the %d bytes "
            "reserved for zip metadata (%d file(s))",
            args.target_zip_mb, safety_margin, len(files),
        )
        return 1

    # Complexity probe pass -- cheap relative to the real shrink pass,
    # run through the same worker pool.
    weights: dict[Path, int] = {}
    probe_failures = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(logging.DEBUG if args.verbose else logging.INFO,),
    ) as executor:
        futures = {executor.submit(_probe_one, p): p for p in files}
        for future in as_completed(futures):
            path = futures[future]
            try:
                _, weight, err = future.result()
            except Exception as exc:  # e.g. worker killed by the OS
                weight, err = 0, f"complexity probe failed for '{path}': {exc!r}"
            if err or weight <= 0:
                if err:
                    logger.debug("%s -- falling back to file size as weight", err)
                weights[path] = file_sizes[path]
                probe_failures += 1
            else:
                weights[path] = weight

    if probe_failures:
        logger.info(
            "%d/%d complexity probes fell back to file-size weighting",
            probe_failures, len(files),
        )

    per_file_targets = compute_fair_share_targets(file_sizes, weights, total_budget)

    logger.info(
        "%d files, %d bytes (%.1f MB) currently, packing to fit <= %.1f MB zip",
        len(files), current_total, current_total / 1_000_000, args.target_zip_mb,
    )
    # Only JPEGs can be copied as-is; everything else is always re-encoded.
    already_fitting = sum(
        1 for p in files
        if is_native_jpeg(p) and per_file_targets[p] == file_sizes[p]
    )
    logger.info(
        "%d JPEG(s) already within their complexity-weighted fair share and "
        "won't be re-encoded; %d file(s) will be shrunk or converted",
        already_fitting, len(files) - already_fitting,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    with tempfile.TemporaryDirectory(prefix="shrink_jpegs_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        # Index-based temp names, not zip names: on a case-insensitive
        # filesystem two distinct zip names could map to the same temp file.
        tmp_dsts = {src: tmp_dir / f"{i}.jpg" for i, src in enumerate(files)}
        successes: list[Path] = []  # sources whose shrunk copy goes in the zip

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(logging.DEBUG if args.verbose else logging.INFO,),
        ) as executor:
            future_to_src = {
                executor.submit(
                    _process_one, src, tmp_dsts[src], per_file_targets[src],
                ): src
                for src in files
            }
            for future in as_completed(future_to_src):
                src = future_to_src[future]
                try:
                    ok, _size, err = future.result()
                except Exception as exc:  # e.g. worker killed by the OS
                    ok, err = False, f"failed to process '{src}': {exc!r}"
                if err:
                    logger.error("%s", err)
                    failures += 1
                elif not ok:
                    logger.warning(
                        "'%s' will be EXCLUDED from the zip -- couldn't hit its "
                        "target size, including it would break the size guarantee",
                        src,
                    )
                    failures += 1
                else:
                    successes.append(src)

        if not successes:
            logger.error("no images were successfully shrunk -- nothing to zip")
            return 1

        # Write under a temp name and rename into place, so a failed or
        # interrupted run never leaves a truncated zip at the output path.
        tmp_zip = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
        try:
            with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_STORED) as zf:
                for src in sorted(successes, key=arcnames.__getitem__):
                    zf.write(tmp_dsts[src], arcname=arcnames[src])
            tmp_zip.replace(args.output)
        finally:
            tmp_zip.unlink(missing_ok=True)

    final_zip_size = args.output.stat().st_size
    logger.info(
        "wrote '%s': %d bytes (%.1f MB), %d/%d images included",
        args.output, final_zip_size, final_zip_size / 1_000_000,
        len(successes), len(files),
    )
    if final_zip_size > target_zip_bytes:
        # Every included file met its target and the metadata reserve is an
        # upper bound, so this means a source changed mid-run or a bug.
        logger.error(
            "final zip (%d bytes) is over the %d byte target",
            final_zip_size, target_zip_bytes,
        )
        return 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
