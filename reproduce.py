"""Rapordaki tüm sonuçları sırayla yeniden üretir.

Gereksinimler: Python ortamı (requirements.txt), Java 17+, bash (Windows'ta Git Bash) ve proje kökünde
Db1_a, Db1_b, Db2_*, Db3_*, Db4_* klasörleri (README'ye bakın).

Kullanım:
    python reproduce.py                 # her şey, ince ayar dahil (CPU'da ~1,5–2 saat)
    python reproduce.py --skip-train    # mevcut runs/arcface/best.pt ile (indirme linki README'de)
    python reproduce.py --only eval     # yalnızca skor dosyalarından metrik ve şekiller
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
SEP = ";" if sys.platform == "win32" else ":"
CP = f"build{SEP}tools/lib/*"
WEIGHTS = "runs/arcface/best.pt"
TUNE_DBS = ["Db1_b", "Db2_b", "Db3_b", "Db4_b"]
DPI = {"Db2_b": "569"}


def run(*cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    t = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"  ({time.perf_counter() - t:.0f} s)", flush=True)


def phase3():
    run(PY, "src/phase3_stats.py")
    run(PY, "src/phase3_figures.py")


def minutiae():
    if not any((ROOT / "tools/lib").glob("sourceafis-*.jar")):
        run("bash", "tools/get_sourceafis.sh")
    run("javac", "-encoding", "UTF-8", "-cp", "tools/lib/*", "-d", "build",
        "src/minutiae/FvcScores.java", "src/minutiae/MatchPair.java")
    for db in ("Db1_a", "Db1_b"):
        run("java", "-cp", CP, "FvcScores", db, f"results/minutiae/scores_sourceafis_{db}.csv")
    for db in TUNE_DBS:  # eşik seçimi: farklı parmakların tüm izlenimleri impostor
        run("java", "-cp", CP, "FvcScores", db, f"results/tune/scores_sourceafis_{db}.csv", "all", DPI.get(db, "500"))


def deep(skip_train):
    for inp in ("raw", "gabor"):  # ince ayarsız ImageNet taban çizgisi
        for db in ("Db1_a", "Db1_b"):
            run(PY, "src/deep/embed_scores.py", db, "--input", inp)
        for db in TUNE_DBS:
            run(PY, "src/deep/embed_scores.py", db, "--input", inp, "--out", "results/tune",
                "--pairs", f"results/tune/scores_sourceafis_{db}.csv")
    if not (skip_train and (ROOT / WEIGHTS).exists()):
        run(PY, "src/deep/finetune.py", "--epochs", "15", "--out", "runs/arcface")
        (ROOT / "results/deep").mkdir(parents=True, exist_ok=True)
        (ROOT / "results/deep/train_log_arcface.csv").write_bytes((ROOT / "runs/arcface/log.csv").read_bytes())
    for db in ("Db1_a", "Db1_b"):
        run(PY, "src/deep/embed_scores.py", db, "--input", "gabor", "--weights", WEIGHTS, "--name", "resnet18_arcface")
    for db in TUNE_DBS:
        run(PY, "src/deep/embed_scores.py", db, "--input", "gabor", "--weights", WEIGHTS, "--name", "resnet18_arcface",
            "--out", "results/tune", "--pairs", f"results/tune/scores_sourceafis_{db}.csv")


def evaluate():
    systems = [("sourceafis", "SourceAFIS", "results/minutiae"),
               ("resnet18_raw", "ResNet-18 ImageNet raw", "results/deep"),
               ("resnet18_gabor", "ResNet-18 ImageNet gabor", "results/deep"),
               ("resnet18_arcface", "ResNet-18 ArcFace", "results/deep")]
    for tag, name, d in systems:
        t = [f"results/tune/scores_{tag}_{db}.csv" for db in TUNE_DBS]
        run(PY, "src/evaluate.py", f"{d}/scores_{tag}_Db1_a.csv", "--name", name, "--out", d,
            "--tune", f"DB1-B={t[0]}", "--tune", "DB1-4 B=" + ",".join(t))
    comp = ["SourceAFIS=results/minutiae/scores_sourceafis_Db1_a.csv",
            "ResNet-18 ArcFace (ince ayar)=results/deep/scores_resnet18_arcface_Db1_a.csv",
            "ResNet-18 ImageNet (Gabor)=results/deep/scores_resnet18_gabor_Db1_a.csv"]
    run(PY, "src/compare.py", "--out", "results/compare", *comp)
    run(PY, "src/score_hist.py", "--out", "results/compare", *comp)
    run(PY, "src/error_analysis.py", "--out", "results/compare")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true", help="mevcut ağırlıklar varsa ince ayarı atla")
    ap.add_argument("--only", choices=["phase3", "minutiae", "deep", "eval"])
    a = ap.parse_args()
    for d in ("results/minutiae", "results/deep", "results/tune", "results/compare"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    steps = {"phase3": phase3, "minutiae": minutiae, "deep": lambda: deep(a.skip_train), "eval": evaluate}
    t0 = time.perf_counter()
    for name, fn in steps.items():
        if a.only in (None, name):
            print(f"\n===== {name} =====", flush=True)
            fn()
    print(f"\nTamamlandı ({(time.perf_counter() - t0) / 60:.1f} dk).")


if __name__ == "__main__":
    main()
