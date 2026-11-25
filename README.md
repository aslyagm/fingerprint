# Parmak İzi Doğrulama: Minutiae ve Derin Öznitelikler

BİL462 Biyometrik Tanıma dersi projesi (A1). Aynı veri ve protokol üzerinde minutiae tabanlı
klasik bir eşleştirici (SourceAFIS) ile CNN gömmesini (ResNet-18) karşılaştırır: DET eğrisi,
EER ve sabit FMR'de FNMR.

## Sonuçlar (FVC2002 DB1-A, 2800 genuine / 4950 impostor)

| Sistem | EER (%95 GA) | FNMR @ FMR %1 | FNMR @ FMR %0,1* |
|---|---|---|---|
| SourceAFIS 3.18.1 | %2,57 [1,83 – 3,38] | %3,43 | %5,14 |
| ResNet-18 + ArcFace (DB2–DB4 ile ince ayar) | %6,64 [5,68 – 7,81] | %24,21 | %49,68 |
| ResNet-18 ImageNet (ince ayarsız, Gabor girdi) | %36,35 | %80,32 | %89,89 |

Güven aralıkları parmaklar üzerinden 1000 bootstrap ile hesaplanır. \*4950 impostor ile FMR %0,1
yaklaşık 5 yanlış kabule karşılık gelir; bu nokta sınırlı güvenilirliktedir.
Referans: FVC2002 DB1-A resmi sonuçları (31 algoritma; en iyi PA15 %0,10, medyan PA02 %2,57),
[`results/reference/`](results/reference/).

## Veri

FVC2002 lisansı yeniden dağıtıma izin vermediği için veri repoda **yoktur**.

- <http://bias.csr.unibo.it/fvc2002/download.asp> yalnızca B kümelerini (10 parmak × 8) sunar.
- Tam veritabanları (A + B), Maltoni ve ark., *Handbook of Fingerprint Recognition* (3. baskı,
  Springer 2022) 4. bölüm ekindeki zip dosyasındadır (indirme sayfasındaki bağlantı).
  Zip içindeki `FVC2002/Dbs/` altından şu klasörleri proje köküne çıkarın:

```
Db1_a/  Db1_b/                 test (100 parmak) ve aynı sensörden ayrık 10 parmak
Db2_a/  Db2_b/  Db3_a/  Db3_b/  Db4_a/  Db4_b/   CNN ince ayarı ve eşik seçimi
```

FVC2002'de 90 gönüllü 30'ar kişilik üç gruba ayrılıp her grup ayrı bir veritabanına atanmıştır;
DB4 sentetiktir (SFinGe). Bu nedenle DB2–DB4 ile DB1 arasında ortak kişi yoktur.

## Kurulum

- Python 3.13, Java 17+ (SourceAFIS için), bash (Windows'ta Git Bash).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
bash tools/get_sourceafis.sh                               # SourceAFIS jar'ları -> tools/lib/
```

GPU gerekmez; PyTorch CPU sürümü için `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

**İnce ayarlı model ağırlıkları** (`best.pt`, 45 MB): indirme bağlantısı eklenecek.
`runs/arcface/best.pt` olarak kaydedin.

## Sonuçları yeniden üretme

```bash
python reproduce.py --skip-train     # mevcut ağırlıklarla (~25 dk, CPU)
python reproduce.py                  # ince ayar dahil (~1,5–2 saat, CPU)
python reproduce.py --only eval      # yalnızca skor dosyalarından metrik ve şekiller
```

## Demo

```bash
python src/demo.py Db1_a/12_1.tif Db1_a/12_5.tif           # aynı parmak
python src/demo.py Db1_a/12_1.tif Db1_a/47_1.tif --show    # farklı parmak, görüntülerle
```

İki yöntemin skorunu ve kararını yazdırır. Eşikler test verisinden ayrık DB1-4 B kümesinde FMR %1
hedefiyle seçilmiştir.

## Yapı

| Yol | İçerik |
|---|---|
| `reproduce.py` | Tüm sonuçları sırayla üretir |
| `src/preprocess.py` | Segmentasyon, yön alanı, Gabor iyileştirme, inceltme, minutiae çıkarımı |
| `src/phase3_stats.py`, `src/phase3_figures.py` | Görüntü kalite ölçüleri, red kuralları, Faz 3 şekilleri |
| `src/minutiae/FvcScores.java` | SourceAFIS skorları (FVC protokolü veya tüm impostor çiftleri) |
| `src/minutiae/MatchPair.java` | Tek çift için SourceAFIS skoru (demo) |
| `src/deep/embed_scores.py` | ResNet-18 gömmesi + kosinüs benzerliği, SourceAFIS ile aynı çiftler |
| `src/deep/finetune.py` | DB2–DB4 üzerinde ArcFace ince ayarı (doğrulama: Db2–4_b) |
| `src/evaluate.py` | EER, FNMR@FMR, bootstrap GA, DET, ayrık kümede eşik seçimi |
| `src/compare.py`, `src/score_hist.py` | Ortak DET grafiği ve skor dağılımları |
| `src/error_analysis.py` | İki yöntemin hatalarının örtüşmesi, kalite/izlenim kırılımı |
| `src/demo.py` | İki görüntüyü iki yöntemle karşılaştırır |
| `tools/get_sourceafis.sh` | SourceAFIS ve bağımlılıklarını SHA-1 doğrulamalı indirir |
| `results/minutiae/`, `results/deep/` | Skor CSV'leri, metrikler (`metrics_*.json`), DET şekilleri, eğitim günlüğü |
| `results/tune/` | Eşik seçimi kümesi skorları (Db1–4_b) |
| `results/compare/` | Ortak DET, skor dağılımları, hata analizi |
| `results/reference/` | FVC2002 DB1-A resmi sonuçları |
| `results/phase3/` | Kalite analizi grafikleri, CSV'ler, `summary.json` |

Parmak izi görüntüsü içeren şekiller (Faz 3 fig1, fig2, fig5–fig7) FVC lisansı gereği repoda
bulunmaz; `phase3_figures.py` veriyle birlikte çalıştırıldığında yerelde üretilir.

## Değerlendirme protokolü

- Test: yalnızca DB1-A, FVC protokolü — 2800 genuine (100 × C(8,2)), 4950 impostor (C(100,2)).
- CNN ince ayarı DB2–DB4 A kümelerinde (300 parmak), epoch seçimi Db2–4_b üzerinde yapılır.
- Eşikler test verisinden ayrık B kümelerinde seçilir: DB1-B (aynı sensör) ve DB1–4 B birleşimi
  (1120 genuine, farklı parmakların tüm izlenimleriyle 11.520 impostor).
- Kalite kurallarıyla işaretlenen görüntüler testten atılmaz; yalnızca hata analizinde kullanılır.
