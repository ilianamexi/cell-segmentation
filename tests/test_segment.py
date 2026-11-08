"""Synthetic image with known nuclei (no external data)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_nuclei import features, segment  # noqa: E402


def synthetic(n_side=4, radius=30, gap=90, touching_pair=True):
    h = w = gap * (n_side + 1)
    yy, xx = np.mgrid[:h, :w]
    img = 0.1 + 0.15 * xx / w  # uneven background, like the real image
    centers = [(gap * (i + 1), gap * (j + 1)) for i in range(n_side) for j in range(n_side)]
    if touching_pair:  # two nuclei whose disks overlap slightly
        centers[0] = (gap, gap - 26)
        centers.append((gap, gap + 26))
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] += 0.5
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1), len(centers)


def test_counts_nuclei_including_touching_pair():
    img, n = synthetic()
    labels, _ = segment(img, tophat_radius=50, min_distance=20, min_area=300)
    assert labels.max() == n


def test_small_tophat_radius_fragments_or_shrinks_nuclei():
    img, _ = synthetic(touching_pair=False)
    big, _ = segment(img, tophat_radius=50)
    small, _ = segment(img, tophat_radius=15)
    # with a disk smaller than the nuclei, the segmented area collapses
    assert (small > 0).sum() < 0.5 * (big > 0).sum()


def test_features_table():
    img, n = synthetic(touching_pair=False)
    labels, _ = segment(img)
    df = features(labels, img)
    assert len(df) == n and (df["area"] > 2000).all() and (df["solidity"] > 0.9).all()
