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
Only genuine .jpg/.jpeg source files get the cheap path for this --
Image.draft() has libjpeg decode directly at a reduced scale, skipping
a full decode. Every other format (PNG, BMP, GIF, TIFF, WEBP, HEIC)
doesn't support draft-mode scaled decoding, so the probe does a full
decode followed by an explicit resize -- slower per file, but correct.

Transparency: images with an alpha channel are composited onto a white
background before conversion to RGB. A plain mode conversion to RGB
would just discard the alpha channel and keep whatever RGB values sit
underneath it, which for many PNGs/GIFs are garbage or black -- that
produces a visibly wrong result rather than the sane white-background
flattening most people expect from a JPEG conversion.

Animation: GIF, animated WEBP, and multi-page TIFF are flattened to
their first frame -- JPEG has no concept of animation. This is logged
per file so it isn't a silent surprise.

Each image is shrunk (quality first, then dimensions) in parallel
worker processes, and the results are packed into shrunk_images.zip
using ZIP_STORED -- the source data is either already-compressed
(JPEG/HEIC) or about to be freshly JPEG-encoded, so DEFLATE would
spend CPU for no benefit either way.

Non-destructive: originals are never modified. Shrunk copies are
produced in a temp directory and packed into the zip; the temp
directory is cleaned up automatically.

Dependency note: HEIC decoding is not built into Pillow. This script
requires the `pillow-heif` package (`pip install pillow-heif`) only for
.heic/.heif input -- the other formats are handled by Pillow core. If
you're targeting an air-gapped host, vendor that wheel ahead of time --
this script does not fetch it for you.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False

logger = logging.getLogger("shrink_jpegs")

MIN_QUALITY = 20
QUALITY_STEP = 5
DIM_SCALE_FACTOR = 0.9
MAX_DIM_ITERATIONS = 10

JPEG_SUFFIXES = (".jpg", ".jpeg")
HEIF_SUFFIXES = (".heic", ".heif")
OTHER_RASTER_SUFFIXES = (".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
ALL_SUFFIXES = JPEG_SUFFIXES + HEIF_SUFFIXES + OTHER_RASTER_SUFFIXES

PROBE_MAX_DIM = 400
PROBE_QUALITY = 75


def is_native_jpeg(path: Path) -> bool:
    """True only for genuine .jpg/.jpeg source files.

    This is the one format that gets the draft-mode fast decode and the
    copy-unchanged fast path -- every other input format always needs a
    real re-encode since the output is always JPEG.
    """
    return path.suffix.lower() in JPEG_SUFFIXES


def load_as_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow image mode to RGB, compositing transparency onto white.

    img.convert("RGB") on a transparent source just discards the alpha
    channel and keeps whatever RGB values sit underneath it -- for many
    PNGs/GIFs those pixels are garbage colors (often black), producing a
    visibly wrong result. Compositing onto white first gives a sane,
    predictable flattening instead.
    """
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if not has_alpha:
        return img.convert("RGB")

    img = img.convert("RGBA")
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[-1])
    return background


def estimate_complexity_weight(path: Path) -> tuple[int, str | None]:
    """Estimate a fair-share weight for one image based on encode complexity.

    For genuine JPEG input, Image.draft() has libjpeg decode directly at
    a reduced scale (1/2, 1/4, or 1/8) instead of a full decode -- this
    is what keeps the probe cheap relative to the real shrink pass. All
    other formats (PNG, BMP, GIF, TIFF, WEBP, HEIC) don't support
    draft-mode scaled decoding, so they get a full decode followed by an
    explicit thumbnail resize -- slower per file, but the estimate is
    still correct. The scaled-down image is encoded once at a fixed
    quality (not searched), and the resulting byte count is scaled up
    by (original_pixels / thumbnail_pixels) to estimate "bytes this
    image would need at a fixed quality, at full resolution."

    Returns (weight_bytes, error_or_None). Callers should fall back to
    on-disk file size as the weight if error is not None.
    """
    try:
        img = Image.open(path)
        native_jpeg = is_native_jpeg(path)
        original_w, original_h = img.size  # header read only, no decode yet
        original_pixels = original_w * original_h
        if original_pixels == 0:
            raise ValueError("zero-pixel image")

        if native_jpeg:
            img.draft("RGB", (PROBE_MAX_DIM, PROBE_MAX_DIM))
        img = load_as_rgb(img)
        if not native_jpeg:
            img.thumbnail((PROBE_MAX_DIM, PROBE_MAX_DIM), Image.LANCZOS)

        thumb_pixels = img.size[0] * img.size[1]
        if thumb_pixels == 0:
            raise ValueError("thumbnail collapsed to zero pixels")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=PROBE_QUALITY)
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


def compress_to_target(
    src_path: Path, final_dst_path: Path, target_bytes: int
) -> tuple[bool, int]:
    """Compress/convert a single image to a JPEG under target_bytes.

    Always writes to a temp file next to final_dst_path first, then
    renames into place on success -- this is safe even when
    final_dst_path == src_path, since we never truncate the source
    file we're still reading pixel data from mid-loop.

    Returns (success, final_size_bytes). final_dst_path is only
    written if a passing result is found.
    """
    native_jpeg = is_native_jpeg(src_path)

    try:
        img = Image.open(src_path)
        n_frames = getattr(img, "n_frames", 1)
        if n_frames > 1:
            logger.warning(
                "'%s' has %d frames -- using the first frame only, "
                "JPEG output can't preserve animation",
                src_path, n_frames,
            )
        img = load_as_rgb(img)
    except Exception as exc:
        raise OSError(f"failed to open '{src_path}': {exc}") from exc

    original_size = src_path.stat().st_size
    # Only genuine JPEG input gets the raw-copy fast path -- every other
    # format must always be re-encoded regardless of its original size,
    # since the output format itself is changing.
    if native_jpeg and original_size <= target_bytes:
        logger.info(
            "'%s' already %d bytes (<= target %d), copying unchanged",
            src_path, original_size, target_bytes,
        )
        if final_dst_path != src_path:
            final_dst_path.write_bytes(src_path.read_bytes())
        return True, original_size

    tmp_path = final_dst_path.with_suffix(final_dst_path.suffix + ".tmp_shrink")
    current_img = img
    dim_iteration = 0

    try:
        while dim_iteration <= MAX_DIM_ITERATIONS:
            quality = 95
            while quality >= MIN_QUALITY:
                current_img.save(tmp_path, format="JPEG", quality=quality, optimize=True)
                size = tmp_path.stat().st_size
                if size <= target_bytes:
                    tmp_path.replace(final_dst_path)
                    logger.info(
                        "'%s' -> %d bytes at quality=%d, dims=%s",
                        src_path, size, quality, current_img.size,
                    )
                    return True, size
                quality -= QUALITY_STEP

            # Quality alone didn't get there; downscale dimensions and retry.
            dim_iteration += 1
            new_w = int(current_img.width * DIM_SCALE_FACTOR)
            new_h = int(current_img.height * DIM_SCALE_FACTOR)
            if new_w < 50 or new_h < 50:
                break
            current_img = current_img.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(
                "'%s' still over target after quality pass, downscaling to %dx%d",
                src_path, new_w, new_h,
            )

        # Failed to hit target even at min quality + min dimensions.
        final_size = tmp_path.stat().st_size if tmp_path.exists() else original_size
        logger.error(
            "'%s' could not be shrunk below %d bytes (best: %d bytes at min quality/dims)",
            src_path, target_bytes, final_size,
        )
        return False, final_size
    finally:
        tmp_path.unlink(missing_ok=True)


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
        "--workers", type=int, default=50,
        help=(
            "Max images to process concurrently (default: 50). This is "
            "CPU-bound work -- throughput plateaus at your core count "
            "(detected and logged at startup); workers beyond that just "
            "use extra memory without going faster."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input_dir.is_dir():
        logger.error("input_dir '%s' is not a directory", args.input_dir)
        return 1

    pattern = "**/*" if args.recursive else "*"
    files = [
        p for p in args.input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in ALL_SUFFIXES
    ]

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

    cpu_count = os.cpu_count() or 1
    workers = max(1, min(args.workers, len(files)))
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
            path, weight, err = future.result()
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

    target_zip_bytes = int(args.target_zip_mb * 1_000_000)
    # Reserve headroom for zip metadata (local headers, central directory)
    # -- small per file, but scales with file count.
    safety_margin = max(50_000, int(target_zip_bytes * 0.01))
    total_budget = target_zip_bytes - safety_margin

    per_file_targets = compute_fair_share_targets(file_sizes, weights, total_budget)

    logger.info(
        "%d files, %d bytes (%.1f MB) currently, packing to fit <= %.1f MB zip",
        len(files), current_total, current_total / 1_000_000, args.target_zip_mb,
    )
    already_fitting = sum(1 for p in files if per_file_targets[p] == file_sizes[p])
    logger.info(
        "%d file(s) already within their complexity-weighted fair share and "
        "won't be re-encoded; %d will be shrunk",
        already_fitting, len(files) - already_fitting,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    failures = 0
    with tempfile.TemporaryDirectory(prefix="shrink_jpegs_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # (src_path, tmp_dst_path, arcname) for every file. Arcnames are
        # deduplicated: multiple source formats can share a stem (e.g.
        # photo.bmp and photo.webp both naively becoming photo.jpg), so
        # only disambiguate on an actual collision -- keeps the common
        # single-format case's output names clean.
        used_arcnames: set[Path] = set()
        tasks: list[tuple[Path, Path, Path]] = []
        for src in files:
            rel = src.relative_to(args.input_dir)
            arcname = rel if is_native_jpeg(src) else rel.with_suffix(".jpg")

            if arcname in used_arcnames:
                disambiguated = arcname.parent / f"{rel.name}.jpg"
                if disambiguated in used_arcnames:
                    i = 2
                    while disambiguated in used_arcnames:
                        disambiguated = arcname.parent / f"{rel.name}_{i}.jpg"
                        i += 1
                logger.warning(
                    "'%s' -- name collision in the zip against another file's "
                    "'%s', writing as '%s' instead",
                    src, arcname, disambiguated,
                )
                arcname = disambiguated

            used_arcnames.add(arcname)
            tmp_dst = tmp_dir / arcname
            tmp_dst.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((src, tmp_dst, arcname))

        successes: list[Path] = []  # tmp paths to zip up

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(logging.DEBUG if args.verbose else logging.INFO,),
        ) as executor:
            future_to_task = {
                executor.submit(_process_one, src, tmp_dst, per_file_targets[src]): (
                    src, tmp_dst,
                )
                for src, tmp_dst, _arcname in tasks
            }
            for future in as_completed(future_to_task):
                src, tmp_dst = future_to_task[future]
                ok, _size, err = future.result()
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
                    successes.append(tmp_dst)

        if not successes:
            logger.error("no images were successfully shrunk -- nothing to zip")
            return 1

        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as zf:
            for tmp_dst in successes:
                arcname = tmp_dst.relative_to(tmp_dir)
                zf.write(tmp_dst, arcname=arcname)

    final_zip_size = args.output.stat().st_size
    logger.info(
        "wrote '%s': %d bytes (%.1f MB), %d/%d images included",
        args.output, final_zip_size, final_zip_size / 1_000_000,
        len(files) - failures, len(files),
    )
    if final_zip_size > target_zip_bytes:
        logger.warning(
            "final zip (%.1f MB) is over the %.1f MB target -- likely because "
            "some images couldn't be shrunk enough at minimum quality/dimensions",
            final_zip_size / 1_000_000, args.target_zip_mb,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
