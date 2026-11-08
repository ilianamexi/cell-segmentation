"""Nucleus segmentation in fluorescence microscopy with top-hat filtering and marker-based watershed.

Pipeline
1. Light Gaussian smoothing (sigma = 2 px) to suppress pixel noise.
2. White top-hat with a disk LARGER than a nucleus (radius 50 px; nuclei here are about 65 px across)
   to remove the uneven background illumination. A disk smaller than the objects would erase their
   interiors and leave only fragments, which is what radius 15 did in the first version.
3. Otsu threshold, removal of small specks, hole filling and a small opening.
4. Distance transform; its smoothed maxima (at least 20 px apart) become one marker per nucleus.
5. Watershed on the inverted distance map splits touching nuclei; tiny regions are dropped.
6. Shape features per nucleus: area, perimeter, axis ratio, eccentricity, solidity, mean intensity.

Usage:
    python segment_nuclei.py                              # default image, writes results/ and figures/
    python segment_nuclei.py --image my.png --tophat-radius 40
    python segment_nuclei.py --compare-radius 15          # also draw the old setting for comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import color, filters, io, measure, morphology, segmentation
from skimage.feature import peak_local_max

ROOT = Path(__file__).resolve().parent


def load_gray(path: Path) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img[..., :3])  # already float in [0, 1]
    else:
        img = img.astype(float) / np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else img.astype(float)
    return img


def segment(gray: np.ndarray, tophat_radius: int = 50, sigma: float = 2.0, min_distance: int = 20,
            min_area: int = 300) -> tuple[np.ndarray, dict]:
    smooth = filters.gaussian(gray, sigma)
    tophat = morphology.white_tophat(smooth, morphology.disk(tophat_radius))
    threshold = filters.threshold_otsu(tophat)
    mask = tophat > threshold
    mask = morphology.remove_small_objects(mask, max_size=min_area // 2)
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.binary_opening(mask, morphology.disk(3))

    distance = filters.gaussian(ndi.distance_transform_edt(mask), 2)
    peaks = peak_local_max(distance, labels=measure.label(mask), min_distance=min_distance, exclude_border=False)
    markers = np.zeros(mask.shape, dtype=int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = segmentation.watershed(-distance, markers, mask=mask)
    labels = morphology.remove_small_objects(labels, max_size=min_area)
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels, {"tophat": tophat, "threshold": float(threshold)}


def features(labels: np.ndarray, gray: np.ndarray) -> pd.DataFrame:
    props = measure.regionprops_table(
        labels, intensity_image=gray,
        properties=["label", "centroid", "area", "perimeter", "major_axis_length", "minor_axis_length",
                    "eccentricity", "solidity", "intensity_mean"])
    df = pd.DataFrame(props).rename(columns={"label": "nucleus_id", "centroid-0": "y", "centroid-1": "x",
                                             "intensity_mean": "mean_intensity"})
    df["axis_ratio"] = df["major_axis_length"] / df["minor_axis_length"].replace(0, np.nan)
    return df.round(3)


def draw(gray, labels, df, path: Path, title: str, old_labels=None, old_title=""):
    panels = 3 if old_labels is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(6.2 * panels, 5.4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Input image")
    if old_labels is not None:
        axes[1].imshow(color.label2rgb(old_labels, image=gray, bg_label=0, alpha=0.45))
        axes[1].set_title(old_title)
    ax = axes[-1]
    ax.imshow(color.label2rgb(labels, image=gray, bg_label=0, alpha=0.35))
    ax.contour(labels > 0, [0.5], colors="#ffd400", linewidths=0.6)
    for r in df.itertuples():
        ax.text(r.x, r.y, str(r.nucleus_id), color="white", fontsize=6.5, ha="center", va="center")
    ax.set_title(title)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default=str(ROOT / "normal40_var_refl.png"))
    p.add_argument("--tophat-radius", type=int, default=50)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--min-distance", type=int, default=20)
    p.add_argument("--min-area", type=int, default=300)
    p.add_argument("--compare-radius", type=int, default=None,
                   help="also segment with this top-hat radius and show it next to the result")
    args = p.parse_args()

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    gray = load_gray(Path(args.image))
    labels, info = segment(gray, args.tophat_radius, args.sigma, args.min_distance, args.min_area)
    df = features(labels, gray)
    df.to_csv(ROOT / "results" / "nuclei_features.csv", index=False)

    old = None
    old_title = ""
    if args.compare_radius:
        old, _ = segment(gray, args.compare_radius, args.sigma, args.min_distance, args.min_area)
        old_title = f"Top-hat radius {args.compare_radius}: {old.max()} objects"
    draw(gray, labels, df, ROOT / "figures" / "segmentation.png",
         f"Top-hat radius {args.tophat_radius}: {len(df)} nuclei", old, old_title)

    summary = df[["area", "perimeter", "axis_ratio", "eccentricity", "solidity"]].describe().T
    summary = summary[["mean", "std", "min", "50%", "max"]].round(2)
    print(f"{len(df)} nuclei segmented (Otsu threshold on top-hat = {info['threshold']:.4f})")
    print(summary.to_string())
    (ROOT / "results" / "summary.md").write_text(
        f"{len(df)} nuclei segmented in `{Path(args.image).name}` "
        f"(top-hat radius {args.tophat_radius} px, sigma {args.sigma}, min distance {args.min_distance} px).\n\n"
        + summary.to_markdown() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
