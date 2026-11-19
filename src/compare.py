"""Birden çok sistemin DB1-A skorlarını tek DET grafiğinde ve tek tabloda karşılaştırır.

Kullanım:
    python src/compare.py --out results/compare \
        "SourceAFIS=results/minutiae/scores_sourceafis_Db1_a.csv" \
        "ResNet-18 (ImageNet)=results/deep/scores_resnet18_gabor_Db1_a.csv"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load, metrics, plot_det  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="+", help="AD=skor.csv")
    ap.add_argument("--out", default="results/compare")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    curves, table = {}, []
    for s in a.systems:
        name, path = s.split("=", 1)
        df = load(path)
        gen = df[df.label == 1].score.to_numpy()
        imp = df[df.label == 0].score.to_numpy()
        m = metrics(gen, imp)
        curves[name] = (gen, imp, m)
        table.append({"system": name, **{k: m[k] for k in ("eer", "fnmr@fmr0.01", "fnmr@fmr0.001")}})

    plot_det(curves, out / "det_comparison.png")
    (out / "comparison.json").write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'Sistem':32s} {'EER':>7s} {'FNMR@1%':>8s} {'FNMR@0.1%':>9s}")
    for r in table:
        print(f"{r['system']:32s} {100*r['eer']:6.2f}% {100*r['fnmr@fmr0.01']:7.2f}% {100*r['fnmr@fmr0.001']:8.2f}%")


if __name__ == "__main__":
    main()
