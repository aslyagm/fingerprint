"""Skor dosyalarından doğrulama metrikleri ve DET eğrisi.

Girdi CSV: probe,candidate,label,score  (label 1 = genuine; yüksek skor = daha benzer)
Metrikler: EER, FNMR@FMR=%1 ve %0,1; parmaklar üzerinden bootstrap %95 güven aralığı.

Kullanım:
    python src/evaluate.py results/minutiae/scores_sourceafis_Db1_a.csv --name SourceAFIS \
        --tune "DB1-B=results/tune/scores_sourceafis_Db1_b.csv" --out results/minutiae
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

FMR_TARGETS = (0.01, 0.001)


def load(path):
    df = pd.read_csv(path)
    df["f_probe"] = df.probe.str.split("_").str[0].astype(int)
    df["f_cand"] = df.candidate.str.split("_").str[0].astype(int)
    return df


def rates(gen, imp, w_gen=None, w_imp=None):
    """Tüm eşikler için (eşik, FMR, FNMR). Kabul: skor >= eşik."""
    w_gen = np.ones_like(gen) if w_gen is None else w_gen
    w_imp = np.ones_like(imp) if w_imp is None else w_imp
    thr = np.unique(np.concatenate([gen, imp]))
    thr = np.append(thr, np.inf)
    og, oi = np.argsort(gen), np.argsort(imp)
    gs, gw = gen[og], np.cumsum(w_gen[og])
    is_, iw = imp[oi], np.cumsum(w_imp[oi])
    # eşiğin altında kalan ağırlık
    below_g = np.concatenate([[0], gw])[np.searchsorted(gs, thr, side="left")]
    below_i = np.concatenate([[0], iw])[np.searchsorted(is_, thr, side="left")]
    fnmr = below_g / gw[-1]
    fmr = 1 - below_i / iw[-1]
    return thr, fmr, fnmr


def metrics(gen, imp, w_gen=None, w_imp=None):
    thr, fmr, fnmr = rates(gen, imp, w_gen, w_imp)
    k = np.argmin(np.abs(fmr - fnmr))
    out = {"eer": float((fmr[k] + fnmr[k]) / 2), "eer_threshold": float(thr[k])}
    for t in FMR_TARGETS:
        ok = np.where(fmr <= t)[0]
        j = ok[np.argmin(fnmr[ok])]
        out[f"fnmr@fmr{t:g}"] = float(fnmr[j])
        out[f"thr@fmr{t:g}"] = float(thr[j])
    return out


def bootstrap(df, n_boot=1000, seed=0):
    """Parmakları iadeli yeniden örnekler; çift ağırlığı = parmak sayılarının çarpımı."""
    rng = np.random.default_rng(seed)
    fingers = np.unique(np.concatenate([df.f_probe, df.f_cand]))
    idx = {f: i for i, f in enumerate(fingers)}
    g = df[df.label == 1]
    i = df[df.label == 0]
    gp = g.f_probe.map(idx).to_numpy()
    ip, ic = i.f_probe.map(idx).to_numpy(), i.f_cand.map(idx).to_numpy()
    gs, is_ = g.score.to_numpy(), i.score.to_numpy()
    res = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(fingers), len(fingers)), minlength=len(fingers)).astype(float)
        wg, wi = cnt[gp], cnt[ip] * cnt[ic]
        mg, mi = wg > 0, wi > 0
        res.append(metrics(gs[mg], is_[mi], wg[mg], wi[mi]))
    return pd.DataFrame(res)


def det_axes(ax):
    ticks = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95]
    labels = ["0,01", "0,1", "1", "5", "20", "50", "80", "95"]
    ax.set_xticks(norm.ppf(ticks), labels)
    ax.set_yticks(norm.ppf(ticks), labels)
    ax.set_xlim(norm.ppf(0.0001), norm.ppf(0.95))
    ax.set_ylim(norm.ppf(0.0001), norm.ppf(0.95))
    ax.set_xlabel("FMR (%)")
    ax.set_ylabel("FNMR (%)")
    ax.grid(True, alpha=0.3)


def plot_det(curves, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    eps = 1e-6
    for name, (gen, imp, m) in curves.items():
        _, fmr, fnmr = rates(gen, imp)
        ax.plot(norm.ppf(np.clip(fmr, eps, 1 - eps)), norm.ppf(np.clip(fnmr, eps, 1 - eps)),
                label=f"{name} (EER %{100 * m['eer']:.2f})".replace(".", ","))
        ax.plot(norm.ppf(m["eer"]), norm.ppf(m["eer"]), "o", ms=4, color=ax.lines[-1].get_color())
    ax.plot(norm.ppf([1e-4, 0.95]), norm.ppf([1e-4, 0.95]), ":", color="gray", lw=0.8)
    det_axes(ax)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("DET — FVC2002 DB1-A")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--name", default="sistem")
    ap.add_argument("--tune", action="append",
                    help="eşik seçimi için ayrık skorlar: ETİKET=a.csv,b.csv (tekrarlanabilir)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(a.scores)
    gen = df[df.label == 1].score.to_numpy()
    imp = df[df.label == 0].score.to_numpy()
    m = metrics(gen, imp)
    bs = bootstrap(df, a.n_boot)
    ci = {k: [float(bs[k].quantile(0.025)), float(bs[k].quantile(0.975))]
          for k in ("eer", "fnmr@fmr0.01", "fnmr@fmr0.001")}
    summary = {"name": a.name, "n_genuine": len(gen), "n_impostor": len(imp), **m,
               "ci95": ci, "n_boot": a.n_boot}

    summary["thresholds"] = {}
    for spec in a.tune or []:
        label, paths = spec.split("=", 1) if "=" in spec else ("tune", spec)
        t = pd.concat([load(p) for p in paths.split(",")])
        tg, ti = t[t.label == 1].score.to_numpy(), t[t.label == 0].score.to_numpy()
        tm = metrics(tg, ti)
        points = {"EER": tm["eer_threshold"], "FMR %1": tm["thr@fmr0.01"], "FMR %0,1": tm["thr@fmr0.001"]}
        summary["thresholds"][label] = {
            "n_genuine": len(tg), "n_impostor": len(ti),
            "points": {k: {"threshold": float(v),
                           "test_fmr": float(np.mean(imp >= v)),
                           "test_fnmr": float(np.mean(gen < v))} for k, v in points.items()},
        }

    tag = a.name.lower().replace(" ", "_")
    (out / f"metrics_{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_det({a.name: (gen, imp, m)}, out / f"det_{tag}.png")

    print(f"{a.name}: EER %{100*m['eer']:.2f} [{100*ci['eer'][0]:.2f}, {100*ci['eer'][1]:.2f}]")
    for t in FMR_TARGETS:
        k = f"fnmr@fmr{t:g}"
        print(f"  FNMR @ FMR %{100*t:g}: %{100*m[k]:.2f} [{100*ci[k][0]:.2f}, {100*ci[k][1]:.2f}]")
    for label, s in summary["thresholds"].items():
        print(f"  Eşik seçimi: {label} ({s['n_genuine']} genuine, {s['n_impostor']} impostor)")
        for k, p in s["points"].items():
            print(f"    {k:9s} eşiği {p['threshold']:.3f} -> testte FMR %{100*p['test_fmr']:.2f}, "
                  f"FNMR %{100*p['test_fnmr']:.2f}")


if __name__ == "__main__":
    main()
