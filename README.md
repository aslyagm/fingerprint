# Parmak İzi Doğrulama: Minutiae ve Derin Öznitelikler

BİL462 Biyometrik Tanıma dersi projesi (A1). Aynı veri ve protokol üzerinde minutiae tabanlı
klasik bir eşleştirici ile CNN gömmesini karşılaştırır: DET eğrisi, EER ve sabit FMR'de FNMR.

## Veri

**FVC2002 DB1** (optik sensör, 388×374 px, 500 dpi). Veri lisansı yeniden dağıtıma izin
vermediği için repoda **yoktur**.

1. <http://bias.csr.unibo.it/fvc2002/> adresinden DB1 kümesini indirin.
2. Klasörleri proje köküne koyun:

```
Db1_a/   1_1.tif … 100_8.tif    (test: 100 parmak × 8)
Db1_b/   101_1.tif … 110_8.tif  (eşik/ayar: 10 parmak × 8)
```

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
```

## Çalıştırma

Faz 3 (veri ve ön işleme) sonuçlarını yeniden üretmek için:

```bash
python src/phase3_stats.py      # results/phase3/quality.csv
python src/phase3_figures.py    # şekiller, quality_flagged.csv, summary.json
```

Minutiae dalı (SourceAFIS 3.18.1, Java 17+ gerekir):

```bash
bash tools/get_sourceafis.sh                                   # jar'ları tools/lib/ altına indirir
javac -encoding UTF-8 -cp "tools/lib/*" -d build src/minutiae/FvcScores.java
java -cp "build;tools/lib/*" FvcScores Db1_a results/minutiae/scores_sourceafis_Db1_a.csv   # Linux/macOS: ':'
java -cp "build;tools/lib/*" FvcScores Db1_b results/minutiae/scores_sourceafis_Db1_b.csv
python src/evaluate.py results/minutiae/scores_sourceafis_Db1_a.csv --name SourceAFIS     --tune results/minutiae/scores_sourceafis_Db1_b.csv --out results/minutiae
```

## Yapı

| Yol | İçerik |
|---|---|
| `src/preprocess.py` | Segmentasyon, yön alanı, Gabor iyileştirme, inceltme, minutiae çıkarımı |
| `src/phase3_stats.py` | Görüntü başına kalite ölçüleri |
| `src/phase3_figures.py` | Faz 3 şekilleri, red kuralları ve özet |
| `src/minutiae/FvcScores.java` | FVC protokolüyle SourceAFIS genuine/impostor skorları |
| `src/evaluate.py` | EER, FNMR@FMR, parmak bazlı bootstrap GA, DET eğrisi (her iki dal için) |
| `tools/get_sourceafis.sh` | SourceAFIS ve bağımlılıklarını SHA-1 doğrulamalı indirir |
| `results/minutiae/` | Skor CSV'leri, `metrics_sourceafis.json`, DET şekli |
| `results/phase3/` | Şekiller, CSV'ler, `summary.json`, slayt notları |

## Değerlendirme protokolü

- Test: yalnızca DB1-A, FVC protokolü — 2800 genuine (100 × C(8,2)), 4950 impostor (C(100,2)).
- CNN ince ayarı ve eşik seçimi test parmaklarından ayrık verilerle yapılır.
- Kalite kurallarıyla işaretlenen görüntüler testten atılmaz; yalnızca hata analizinde kullanılır.

## Yapay zekâ kullanımı

Kod ve şekillerin hazırlanmasında Claude (Anthropic) kullanılmıştır; ayrıntılar raporlardaki
Yapay Zekâ Kullanım Beyanı'ndadır.
