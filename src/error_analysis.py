"""SourceAFIS ile ince ayarlı CNN'in DB1-A üzerindeki hatalarının karşılaştırmalı analizi.

Çalışma noktası: her sistem için test verisinden AYRIK eşik seti (DB1-4 B) üzerinde FMR %1 hedefiyle
seçilen eşik (evaluate.py çıktısındaki metrics_*.json). Çıktılar:
    error_analysis.json         hata sayıları, örtüşme, kalite/izlenim kırılımı, skor korelasyonu
    error_scatter.png           iki sistemin skorları (genuine / impostor) ve eşikleri
    hardest_pairs.csv           her sistem için en düşük genuine / en yüksek impostor çiftleri

Kullanım:
    python src/error_analysis.py --out results/compare
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SYSTEMS = {
    "SourceAFIS": ("results/minutiae/scores_sourceafis_Db1_a.csv", "results/minutiae/metrics_sourceafis.json"),
    "CNN": ("results/deep/scores_resnet18_arcface_Db1_a.csv", "results/deep/metrics_resnet-18_arcface.json"),
}
TUNE, POINT = "DB1-4 B", "FMR %1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/compare")
    ap.add_argument("--quality", default="results/phase3/quality_flagged.csv")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df, thr = None, {}
    for name, (scores, metrics) in SYSTEMS.items():
        s = pd.read_csv(scores).rename(columns={"score": name})
        df = s if df is None else df.merge(s, on=["probe", "candidate", "label"])
        thr[name] = json.loads(Path(metrics).read_text(encoding="utf-8"))["thresholds"][TUNE]["points"][POINT]["threshold"]

    q = pd.read_csv(a.quality)
    q = q[q.subset == "Db1_a"]
    flagged = set(q[q.rejected].finger.astype(str) + "_" + q[q.rejected].impression.astype(str))
    df["flagged"] = df.probe.isin(flagged) | df.candidate.isin(flagged)

    gen, imp = df[df.label == 1].copy(), df[df.label == 0].copy()
    for n in SYSTEMS:
        gen[f"{n}_err"] = gen[n] < thr[n]      # yanlış red
        imp[f"{n}_err"] = imp[n] >= thr[n]     # yanlış kabul
    sa, cn = "SourceAFIS_err", "CNN_err"

    def overlap(d):
        return {"yalnız SourceAFIS": int((d[sa] & ~d[cn]).sum()), "yalnız CNN": int((~d[sa] & d[cn]).sum()),
                "ikisi de": int((d[sa] & d[cn]).sum()), "hiçbiri": int((~d[sa] & ~d[cn]).sum())}

    def flagged_share(d, col):
        e = d[d[col]]
        return {"hata sayısı": int(len(e)), "işaretli görüntü içeren oran": round(float(e.flagged.mean()), 3) if len(e) else None}

    # izlenim bazında yanlış red oranı (çiftte o izlenim varsa sayılır)
    imp_rate = {}
    for k in range(1, 9):
        m = gen.probe.str.endswith(f"_{k}") | gen.candidate.str.endswith(f"_{k}")
        imp_rate[k] = {n: round(float(gen.loc[m, f"{n}_err"].mean()), 4) for n in SYSTEMS}

    res = {
        "çalışma noktası": f"{POINT} eşiği, {TUNE} üzerinde seçildi",
        "eşikler": thr,
        "genuine (yanlış red)": {"toplam çift": len(gen), "örtüşme": overlap(gen),
                                 **{n: flagged_share(gen, f"{n}_err") for n in SYSTEMS},
                                 "tüm genuine çiftlerde işaretli oran": round(float(gen.flagged.mean()), 3)},
        "impostor (yanlış kabul)": {"toplam çift": len(imp), "örtüşme": overlap(imp),
                                    **{n: flagged_share(imp, f"{n}_err") for n in SYSTEMS}},
        "izlenim bazında yanlış red oranı": imp_rate,
        "skor korelasyonu (Spearman)": {
            "genuine": round(float(spearmanr(gen.SourceAFIS, gen.CNN).statistic), 3),
            "impostor": round(float(spearmanr(imp.SourceAFIS, imp.CNN).statistic), 3)},
        "SourceAFIS skoru < 1 olan genuine çift": int((gen.SourceAFIS < 1).sum()),
    }
    (out / "error_analysis.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for n in SYSTEMS:
        for _, r in gen.nsmallest(10, n).iterrows():
            rows.append({"sistem": n, "tür": "en düşük genuine", "probe": r.probe, "candidate": r.candidate,
                         "SourceAFIS": r.SourceAFIS, "CNN": r.CNN, "işaretli": r.flagged})
        for _, r in imp.nlargest(10, n).iterrows():
            rows.append({"sistem": n, "tür": "en yüksek impostor", "probe": r.probe, "candidate": r.candidate,
                         "SourceAFIS": r.SourceAFIS, "CNN": r.CNN, "işaretli": r.flagged})
    pd.DataFrame(rows).to_csv(out / "hardest_pairs.csv", index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(6, 5))
    lx = lambda v: np.log1p(np.maximum(v, 0))  # noqa: E731
    ax.scatter(lx(imp.SourceAFIS), imp.CNN, s=4, alpha=0.35, color="#d62728", label="impostor")
    ax.scatter(lx(gen.SourceAFIS), gen.CNN, s=4, alpha=0.35, color="#1f77b4", label="genuine")
    ax.axvline(lx(thr["SourceAFIS"]), color="k", ls="--", lw=1, label="SourceAFIS eşiği")
    ax.axhline(thr["CNN"], color="gray", ls=":", lw=1.3, label="CNN eşiği")
    ax.set_xlabel("SourceAFIS log(1 + skor)")
    ax.set_ylabel("CNN kosinüs benzerliği")
    ax.set_title(f"Skorların birlikte dağılımı ({POINT} eşikleri, {TUNE})")
    ax.legend(fontsize=8, loc="lower right", markerscale=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "error_scatter.png", dpi=200)
    plt.close(fig)

    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
