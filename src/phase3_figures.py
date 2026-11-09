"""Phase 3 figures and summary from results/phase3/quality.csv.

Usage:  python src/phase3_figures.py [data_root]
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load, process  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join("results", "phase3")
df = pd.read_csv(os.path.join(OUT, "quality.csv"))
df["displacement"] = np.hypot(df.centroid_dx, df.centroid_dy)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ----------------------------------------------------------- rejection rule
# Fixed a priori from general fingerprint-quality practice, not tuned on scores.
RULES = {
    "fg_ratio < 0.25 (çok küçük alan)": df.fg_ratio < 0.25,
    "coherence < 0.45 (belirsiz sırt yapısı)": df.coherence < 0.45,
    "n_minutiae < 12 (az minutiae)": df.n_minutiae < 12,
}
df["reject_reason"] = ""
for name, m in RULES.items():
    df.loc[m, "reject_reason"] += np.where(df.loc[m, "reject_reason"] == "", "", "; ") + name
df["rejected"] = df.reject_reason != ""
df.to_csv(os.path.join(OUT, "quality_flagged.csv"), index=False)


def path(f):
    return os.path.join(root, f)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def show_minutiae(ax, img, minutiae):
    ax.imshow(img, cmap="gray")
    for y, x, t in minutiae:
        ax.plot(x, y, "o" if t == 1 else "s", mfc="none", ms=6, mew=1.3,
                mec="#d62728" if t == 1 else "#1f77b4")


# ------------------------------------------------ 1. dataset overview grid
fig, axes = plt.subplots(2, 8, figsize=(18, 5))
for row, finger in enumerate([1, 59]):
    for i in range(8):
        ax = axes[row, i]
        ax.imshow(load(path(f"Db1_a/{finger}_{i + 1}.tif")), cmap="gray")
        ax.set_title(f"{finger}_{i + 1}", fontsize=10)
        ax.axis("off")
fig.suptitle("FVC2002 DB1-A: aynı parmağın 8 görüntüsü (üst: parmak 1, alt: parmak 59)")
save(fig, "fig1_dataset_overview.png")

# --------------------------------------------------- 2. pipeline stages
r = process(load(path("Db1_a/1_1.tif")))
fig, axes = plt.subplots(1, 7, figsize=(22, 4.2))
axes[0].imshow(r["img"], cmap="gray"); axes[0].set_title("1. Ham görüntü")
axes[1].imshow(r["img"], cmap="gray")
axes[1].imshow(np.ma.masked_where(r["mask"], r["mask"]), cmap="autumn", alpha=0.45)
axes[1].set_title("2. Segmentasyon")
axes[2].imshow(r["norm"] * r["mask"], cmap="gray"); axes[2].set_title("3. Normalizasyon")
axes[3].imshow(r["img"], cmap="gray", alpha=0.5)
step = 12
for y in range(step // 2, r["img"].shape[0], step):
    for x in range(step // 2, r["img"].shape[1], step):
        if r["mask"][y, x]:
            t = r["theta"][y, x]
            dx, dy = 5 * np.cos(t), 5 * np.sin(t)
            axes[3].plot([x - dx, x + dx], [y - dy, y + dy], "r-", lw=1)
axes[3].set_title("4. Yön alanı")
axes[4].imshow(r["enh"], cmap="gray"); axes[4].set_title(f"5. Gabor (periyot={1 / r['freq']:.1f} px)")
axes[5].imshow(r["skel"], cmap="gray_r"); axes[5].set_title("6. İnceltme (iskelet)")
show_minutiae(axes[6], r["img"], r["minutiae"])
n_end = sum(m[2] == 1 for m in r["minutiae"])
axes[6].set_title(f"7. Minutiae: {n_end} uç, {len(r['minutiae']) - n_end} çatal")
for a in axes:
    a.axis("off")
save(fig, "fig2_pipeline.png")

# ------------------------------------------------- 3. quality distributions
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
specs = [("fg_ratio", "Ön plan oranı", 0.25), ("coherence", "Ortalama yön tutarlılığı", 0.45),
         ("n_minutiae", "Minutiae sayısı", 12), ("mean_gray", "Ön plan ortalama gri seviye", None)]
for ax, (col, label, thr) in zip(axes, specs):
    ax.hist(df[col], bins=35, color="#4c72b0", alpha=0.85)
    if thr is not None:
        ax.axvline(thr, color="#d62728", ls="--", lw=2, label=f"red eşiği = {thr}")
        ax.legend()
    ax.set_xlabel(label); ax.set_ylabel("görüntü sayısı")
save(fig, "fig3_quality_hist.png")

# ------------------------------------------------- 4. quality by impression
g = df.groupby("impression")[["coherence", "mean_gray", "fg_ratio", "displacement"]].mean()
fig, axes = plt.subplots(1, 4, figsize=(20, 3.8))
titles = ["Yön tutarlılığı", "Ortalama gri (yüksek = kuru/açık)", "Ön plan oranı",
          "Merkez kayması (görüntü oranı)"]
for ax, col, t in zip(axes, g.columns, titles):
    ax.bar(g.index, g[col], color="#4c72b0")
    ax.set_title(t); ax.set_xlabel("görüntü no (1-8)"); ax.set_xticks(range(1, 9))
    lo, hi = g[col].min(), g[col].max()
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.15 * (hi - lo))
save(fig, "fig4_quality_by_impression.png")

# ------------------------------------------------- 5. quality issue examples
a = df[df.subset == "Db1_a"]
examples = [
    ("İyi kalite", a.sort_values("coherence").iloc[-1]),
    ("Kuru / soluk", a[a.mean_gray > 225].sort_values("coherence").iloc[0]),
    ("Islak / koyu", a.sort_values("mean_gray").iloc[0]),
    ("Kaymış / kısmi", a.sort_values("displacement").iloc[-1]),
    ("Küçük temas alanı", a.sort_values("fg_ratio").iloc[0]),
]
fig, axes = plt.subplots(2, len(examples), figsize=(4 * len(examples), 8.5))
for i, (name, row) in enumerate(examples):
    rr = process(load(path(row.file)))
    axes[0, i].imshow(rr["img"], cmap="gray")
    axes[0, i].set_title(f"{name}\n{row.file.split('/')[-1]}  koh={row.coherence:.2f}", fontsize=10)
    show_minutiae(axes[1, i], rr["enh"], rr["minutiae"])
    axes[1, i].set_title(f"Gabor + {len(rr['minutiae'])} minutiae", fontsize=10)
for ax in axes.ravel():
    ax.axis("off")
save(fig, "fig5_quality_issues.png")

# ------------------------------------------------- 6. rejected examples
rej = df[df.rejected].copy()
picks = []
for name in RULES:
    sub = rej[rej.reject_reason.str.contains(name.split(" (")[0], regex=False)]
    if len(sub):
        picks.append(sub.sort_values("coherence").iloc[0])
        if len(sub) > 1:
            picks.append(sub.sort_values("coherence").iloc[-1])
picks = picks[:6]
fig, axes = plt.subplots(2, len(picks), figsize=(3.8 * len(picks), 8.5))
for i, row in enumerate(picks):
    rr = process(load(path(row.file)))
    axes[0, i].imshow(rr["img"], cmap="gray")
    axes[0, i].set_title(row.file.split("/")[-1] + "\n" + row.reject_reason.replace("; ", "\n"),
                         fontsize=9)
    show_minutiae(axes[1, i], rr["enh"], rr["minutiae"])
    axes[1, i].set_title(f"koh={row.coherence:.2f}, alan={row.fg_ratio:.2f}, "
                         f"min={row.n_minutiae}", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
save(fig, "fig6_rejected_examples.png")

# ------------------------------------------------- 7. hardest cases: did preprocessing fail?
hard = [
    ("En küçük alan", df.sort_values("fg_ratio").iloc[0]),
    ("En düşük tutarlılık", df.sort_values("coherence").iloc[0]),
    ("En az minutiae", df.sort_values("n_minutiae").iloc[0]),
    ("En büyük kayma", df.sort_values("displacement").iloc[-1]),
]
fig, axes = plt.subplots(1, len(hard), figsize=(4.2 * len(hard), 4.6))
for ax, (name, row) in zip(axes, hard):
    rr = process(load(path(row.file)))
    show_minutiae(ax, rr["img"], rr["minutiae"])
    ax.contour(rr["mask"], levels=[0.5], colors="#ff7f0e", linewidths=2)
    ax.set_title(f"{name}\n{row.file.split('/')[-1]} · {len(rr['minutiae'])} minutiae", fontsize=11)
    ax.axis("off")
save(fig, "fig7_hardest_cases.png")

failures = {
    "empty_mask": int((df.fg_ratio == 0).sum()),
    "frequency_fallback": int((df.freq_blocks == 0).sum()),
    "zero_minutiae": int((df.n_minutiae == 0).sum()),
    "min_fg_ratio": round(float(df.fg_ratio.min()), 3),
    "min_freq_blocks": int(df.freq_blocks.min()),
    "min_minutiae": int(df.n_minutiae.min()),
}

# ------------------------------------------------- summary
summary = {
    "images": {s: int((df.subset == s).sum()) for s in ["Db1_a", "Db1_b"]},
    "fingers": {s: int(df[df.subset == s].finger.nunique()) for s in ["Db1_a", "Db1_b"]},
    "image_size_raw": [int(df.width.iloc[0]), int(df.height.iloc[0])],
    "right_edge_artifact_images": int((df.artifact_cols > 0).sum()),
    "fvc_comparisons_db1a": {"genuine": 100 * 28, "impostor": 100 * 99 // 2},
    "preprocessing_failures": failures,
    "rejected_total": {s: int(df[(df.subset == s)].rejected.sum()) for s in ["Db1_a", "Db1_b"]},
    "rejected_by_rule": {k: int(m.sum()) for k, m in RULES.items()},
    "rejected_by_impression": df[df.rejected].impression.value_counts().sort_index().to_dict(),
    "means": df[["fg_ratio", "mean_gray", "contrast", "coherence", "ridge_period",
                 "n_minutiae", "n_endings", "n_bifurcations"]].mean().round(3).to_dict(),
    "medians": df[["n_minutiae", "ridge_period"]].median().to_dict(),
}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=int)
print(json.dumps(summary, ensure_ascii=False, indent=2, default=int))
print(df[df.rejected][["file", "reject_reason"]].to_string())
