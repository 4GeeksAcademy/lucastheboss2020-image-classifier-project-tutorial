"""Download and inspect the 4Geeks dogs-vs-cats dataset without training a model."""
from pathlib import Path
import argparse
import collections
import json
import re
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip"
RAW = ROOT / "data" / "raw"
ARCHIVE = RAW / "dogs-vs-cats.zip"
DEST = RAW / "dogs-vs-cats"
REPORTS = ROOT / "reports"


def download_and_extract():
    RAW.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        partial = ARCHIVE.with_suffix(".zip.part")
        print("Downloading course dataset...", flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open("wb") as output:
            expected = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            last_report = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 64 * 1024 * 1024:
                    print(f"Downloaded {downloaded // (1024 * 1024)} MiB", flush=True)
                    last_report = downloaded
        if expected and downloaded != expected:
            raise RuntimeError("Dataset download was incomplete; retry before extracting.")
        if not zipfile.is_zipfile(partial):
            raise RuntimeError("Downloaded response is not a ZIP archive.")
        partial.replace(ARCHIVE)
    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            target = (DEST / member.filename).resolve()
            if not target.is_relative_to(DEST.resolve()):
                raise RuntimeError("Unsafe path in ZIP archive.")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    print("Dataset downloaded and extracted.", flush=True)


def inspect_dataset():
    from PIL import Image, ImageOps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = sorted(p for p in DEST.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    files = [p for p in candidates if "__MACOSX" not in p.parts and not p.name.startswith("._")]
    labeled = [p for p in files if re.fullmatch(r"(cat|dog)\.\d+\.jpg", p.name, re.I)]
    if not labeled:
        raise RuntimeError("No labeled cat/dog images found after extraction.")
    counts = collections.Counter()
    modes = collections.Counter()
    sizes = []
    corrupt = []
    examples = {"cat": [], "dog": []}
    for index, path in enumerate(labeled, 1):
        label = path.name.split(".")[0].lower()
        counts[label] += 1
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                sizes.append(image.size)
                modes[image.mode] += 1
                if len(examples[label]) < 9:
                    examples[label].append(path)
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            corrupt.append({"file": str(path.relative_to(ROOT)), "error": type(error).__name__, "message": str(error)})
        if index % 5000 == 0:
            print(f"Checked {index}/{len(labeled)} labeled images", flush=True)
    figures = REPORTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for label, paths in examples.items():
        if len(paths) != 9:
            raise RuntimeError(f"Fewer than nine valid {label} images.")
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for axis, path in zip(axes.flat, paths):
            with Image.open(path) as image:
                original_size = image.size
                axis.imshow(ImageOps.exif_transpose(image).convert("RGB"))
            axis.set_title(f"{path.name} ({original_size[0]} x {original_size[1]})", fontsize=9)
            axis.axis("off")
        fig.suptitle(f"Nine {label} images — original image proportions", fontsize=15)
        fig.tight_layout()
        fig.savefig(figures / f"{label}s_grid.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    summary = {
        "source_url": URL,
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "image_directory": str(DEST.relative_to(ROOT)),
        "labeled_images": len(labeled),
        "class_counts": dict(sorted(counts.items())),
        "unlabeled_images": len(files) - len(labeled),
        "ignored_metadata_files": len(candidates) - len(files),
        "valid_labeled_images": len(labeled) - len(corrupt),
        "corrupt_count": len(corrupt),
        "corrupt_images": corrupt,
        "image_modes": dict(modes),
        "width_range": [min(w for w, h in sizes), max(w for w, h in sizes)],
        "height_range": [min(h for w, h in sizes), max(h for w, h in sizes)],
        "unique_dimensions": len(set(sizes)),
        "figures": ["reports/figures/dogs_grid.png", "reports/figures/cats_grid.png"],
        "notes": "Original files preserved. No train/validation/test split, resizing, augmentation, or model training performed yet. Exclude listed unreadable images during preprocessing."
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data_summary.json").write_text(json.dumps(summary, indent=2) + chr(10))
    print(json.dumps({k: v for k, v in summary.items() if k != "corrupt_images"}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    download_and_extract()
    if not args.download_only:
        inspect_dataset()
