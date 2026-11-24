"""Genuine / impostor skor dağılımları (DB1-A), her sistem için bir panel.

EER eşiği ve FMR %1 eşiği dikey çizgiyle gösterilir. SourceAFIS skorları çok sağa çarpık olduğundan
log(1 + skor) ölçeğinde çizilir; kosinüs benzerlikleri doğrusal ölçektedir.

Kullanım:
    python src/score_hist.py --out results/compare \
        "SourceAFIS=results/minutiae/scores_sourceafis_Db1_a.csv" \
        "ResNet-18 ArcFace=results/deep/scores_resnet18_arcface_Db1_a.csv"
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load, metrics  # noqa: E402

GEN, IMP = "#1f77b4", "#d62728"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="AD=skor.csv")
    ap.add_argument("--out", default="results/compare")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(a.systems), figsize=(5 * len(a.systems), 3.8), squeeze=False)
    for ax, s in zip(axes[0], a.systems):
        name, path = s.split("=", 1)
        df = load(path)
        gen = df[df.label == 1].score.to_numpy()
        imp = df[df.label == 0].score.to_numpy()
        m = metrics(gen, imp)
        logx = gen.max() > 2  # SourceAFIS gibi sınırsız skorlar
        f = (lambda v: np.log1p(np.maximum(v, 0))) if logx else (lambda v: v)
        g, i = f(gen), f(imp)
        bins = np.linspace(min(g.min(), i.min()), max(g.max(), i.max()), 60)
        ax.hist(i, bins, density=True, alpha=0.55, color=IMP, label=f"impostor (n={len(imp)})")
        ax.hist(g, bins, density=True, alpha=0.55, color=GEN, label=f"genuine (n={len(gen)})")
        ax.axvline(f(m["eer_threshold"]), color="k", ls="--", lw=1,
                   label=f"EER eşiği (EER %{100 * m['eer']:.2f})".replace(".", ","))
        ax.axvline(f(m["thr@fmr0.01"]), color="gray", ls=":", lw=1.2, label="FMR %1 eşiği")
        ax.set_title(name)
        ax.set_xlabel("log(1 + skor)" if logx else "kosinüs benzerliği")
        ax.set_ylabel("yoğunluk")
        ax.legend(fontsize=7.5, loc="best", framealpha=0.9)
        ax.grid(alpha=0.25)
    fig.suptitle("Skor dağılımları — FVC2002 DB1-A", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "score_distributions.png", dpi=200, bbox_inches="tight")
    print(out / "score_distributions.png")


if __name__ == "__main__":
    main()
