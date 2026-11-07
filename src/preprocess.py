"""Classical fingerprint preprocessing pipeline (Hong, Wan & Jain 1998 style).

Stages:
    1. segmentation   - block-variance foreground mask
    2. normalization  - zero mean / unit variance inside the mask
    3. orientation    - gradient-based ridge orientation + coherence
    4. frequency      - ridge frequency from oriented block projections
    5. enhancement    - oriented Gabor filtering
    6. binarize/thin  - ridge skeleton
    7. minutiae       - crossing number + simple false-minutiae removal

`process(img)` runs all stages and returns a dict with every intermediate
result so they can be visualized and used for quality statistics.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

BLOCK = 16


# ------------------------------------------------------------------- load
ARTIFACT_COLS = 4  # FVC2002 DB1 images carry a dark 1-3 px line at the right edge


def load(path: str) -> np.ndarray:
    """Read a grayscale FVC image and remove the right-edge sensor artifact."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :-ARTIFACT_COLS].copy()


def right_edge_artifact_width(img: np.ndarray, max_cols: int = 5) -> int:
    """Width (px) of the dark vertical line at the right edge of a raw image."""
    f = img.astype(np.float32)
    ref = f[:, -max_cols - 6:-max_cols].mean()
    k = 0
    for j in range(1, max_cols + 1):
        if f[:, -j].mean() < ref - 15 and f[:, -j].std() < 40:
            k = j
    return k


def _erode_inside(mask: np.ndarray, radius: int) -> np.ndarray:
    """Erode treating everything outside the image as background."""
    m = np.pad(mask.astype(np.uint8), radius + 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.erode(m, k)[radius + 1:-radius - 1, radius + 1:-radius - 1].astype(bool)


# ---------------------------------------------------------------- 1. segment
def segment(img: np.ndarray, block: int = BLOCK, rel_thresh: float = 0.25):
    """Foreground mask from local standard deviation of gray levels."""
    f = img.astype(np.float32)
    f = (f - f.mean()) / (f.std() + 1e-6)
    mean = cv2.blur(f, (block, block))
    sq = cv2.blur(f * f, (block, block))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    mask = std > rel_thresh * std.max()
    # clean up: close gaps, keep largest component, fill holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block, block))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    lab, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = (lab == (np.argmax(sizes) + 1)).astype(np.uint8)
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool), std


# -------------------------------------------------------------- 2. normalize
def normalize(img: np.ndarray, mask: np.ndarray):
    f = img.astype(np.float32)
    m, s = f[mask].mean(), f[mask].std() + 1e-6
    return (f - m) / s


# ------------------------------------------------------------ 3. orientation
def orientation(norm: np.ndarray, grad_sigma: float = 1.0, block_sigma: float = 7.0):
    """Pixel-wise ridge orientation (radians, [0, pi)) and coherence [0, 1]."""
    g = cv2.GaussianBlur(norm, (0, 0), grad_sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), block_sigma)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), block_sigma)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), block_sigma)
    # gradient direction is perpendicular to ridges -> add pi/2
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy) + np.pi / 2
    theta = np.mod(theta, np.pi)
    coh = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-6)
    return theta, np.clip(coh, 0, 1)


# -------------------------------------------------------------- 4. frequency
def ridge_frequency(norm, theta, mask, block=32, min_wl=4, max_wl=16):
    """Median ridge frequency over foreground blocks (cycles/pixel).

    Each block is rotated so ridges are vertical, the columns are summed,
    and the distance between projection peaks gives the ridge wavelength.
    """
    h, w = norm.shape
    freqs = []
    for r in range(0, h - block, block // 2):
        for c in range(0, w - block, block // 2):
            if mask[r:r + block, c:c + block].mean() < 0.9:
                continue
            blk = norm[r:r + block, c:c + block]
            th = theta[r:r + block, c:c + block]
            ang = 0.5 * np.arctan2(np.sin(2 * th).mean(), np.cos(2 * th).mean())
            M = cv2.getRotationMatrix2D((block / 2, block / 2), np.degrees(ang) - 90, 1)
            rot = cv2.warpAffine(blk, M, (block, block), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            crop = rot[block // 4: 3 * block // 4]
            proj = ndi.uniform_filter1d(crop.sum(axis=0), 3)
            peaks = np.where((proj[1:-1] > proj[:-2]) & (proj[1:-1] >= proj[2:])
                             & (proj[1:-1] > proj.mean()))[0]
            if len(peaks) >= 2:
                wl = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
                if min_wl <= wl <= max_wl:
                    freqs.append(1.0 / wl)
    if not freqs:
        return 1 / 9.0, 0  # typical 500 dpi ridge period ~9 px
    return float(np.median(freqs)), len(freqs)


# ------------------------------------------------------------ 5. enhancement
def gabor_enhance(norm, theta, freq, mask, n_angles=16, kx=0.65, ky=0.65):
    wl = 1.0 / freq
    sigma_x, sigma_y = kx * wl, ky * wl
    size = int(np.ceil(3 * max(sigma_x, sigma_y)))
    yy, xx = np.mgrid[-size:size + 1, -size:size + 1].astype(np.float32)
    angles = np.arange(n_angles) * np.pi / n_angles
    idx = np.round(theta / (np.pi / n_angles)).astype(int) % n_angles
    out = np.zeros_like(norm)
    for i, a in enumerate(angles):
        # kernel oscillates across ridges (perpendicular to ridge direction a)
        xr = xx * np.cos(a + np.pi / 2) + yy * np.sin(a + np.pi / 2)
        yr = -xx * np.sin(a + np.pi / 2) + yy * np.cos(a + np.pi / 2)
        k = np.exp(-0.5 * (xr ** 2 / sigma_x ** 2 + yr ** 2 / sigma_y ** 2)) \
            * np.cos(2 * np.pi * freq * xr)
        k -= k.mean()
        resp = cv2.filter2D(norm, cv2.CV_32F, k)
        sel = idx == i
        out[sel] = resp[sel]
    out[~mask] = 0
    return out


# ------------------------------------------------------- 6. binarize / thin
def binarize_thin(enh, mask):
    ridges = (enh < 0) & mask  # ridges are dark in the input -> negative response
    ridges = ndi.binary_opening(ridges, iterations=1)
    skel = skeletonize(ridges)
    return ridges, skel


# --------------------------------------------------------------- 7. minutiae
def extract_minutiae(skel, mask, border=14, min_dist=8):
    """Crossing-number minutiae. Returns list of (row, col, type) with
    type 1 = ridge ending, 3 = bifurcation."""
    s = skel.astype(np.uint8)
    # 8 neighbours in circular order
    nb = [s[:-2, 1:-1], s[:-2, 2:], s[1:-1, 2:], s[2:, 2:],
          s[2:, 1:-1], s[2:, :-2], s[1:-1, :-2], s[:-2, :-2]]
    cn = np.zeros_like(s[1:-1, 1:-1], dtype=np.int32)
    for i in range(8):
        cn += np.abs(nb[i].astype(np.int32) - nb[(i + 1) % 8].astype(np.int32))
    cn //= 2
    core = s[1:-1, 1:-1].astype(bool)
    inner = _erode_inside(mask, border)[1:-1, 1:-1]
    pts = []
    for t in (1, 3):
        rr, cc = np.where(core & (cn == t) & inner)
        pts += [(r + 1, c + 1, t) for r, c in zip(rr, cc)]
    if not pts:
        return []
    # remove clusters of close minutiae (spurs, short breaks, bridges)
    P = np.array([(r, c) for r, c, _ in pts], dtype=np.float32)
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    keep = d.min(axis=1) >= min_dist
    return [p for p, k in zip(pts, keep) if k]


# ------------------------------------------------------------------ driver
def process(img: np.ndarray) -> dict:
    mask, std = segment(img)
    norm = normalize(img, mask)
    theta, coh = orientation(norm)
    freq, n_freq_blocks = ridge_frequency(norm, theta, mask)
    enh = gabor_enhance(norm, theta, freq, mask)
    ridges, skel = binarize_thin(enh, mask)
    minutiae = extract_minutiae(skel, mask)
    return dict(img=img, mask=mask, local_std=std, norm=norm, theta=theta, coh=coh,
                freq=freq, n_freq_blocks=n_freq_blocks, enh=enh, ridges=ridges,
                skel=skel, minutiae=minutiae)


def quality_features(r: dict) -> dict:
    """Simple per-image quality measures used for the Phase 3 statistics."""
    img, mask, coh = r["img"], r["mask"], r["coh"]
    h, w = img.shape
    fg = img[mask] if mask.any() else img.ravel()
    ys, xs = np.nonzero(mask) if mask.any() else (np.array([h / 2]), np.array([w / 2]))
    inner = _erode_inside(mask, 8)
    n_end = sum(1 for m in r["minutiae"] if m[2] == 1)
    return dict(
        fg_ratio=float(mask.mean()),
        mean_gray=float(fg.mean()),
        contrast=float(fg.std()),
        coherence=float(coh[inner].mean()) if inner.any() else 0.0,
        ridge_period=float(1.0 / r["freq"]),
        freq_blocks=int(r["n_freq_blocks"]),
        n_minutiae=len(r["minutiae"]),
        n_endings=n_end,
        n_bifurcations=len(r["minutiae"]) - n_end,
        centroid_dx=float(xs.mean() / w - 0.5),
        centroid_dy=float(ys.mean() / h - 0.5),
        touches_border=bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
    )
