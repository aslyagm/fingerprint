import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * FVC protokolüne göre SourceAFIS skorları üretir.
 *
 * Genuine: her parmak için aynı parmağın tüm izlenim çiftleri, i < j  -> P * C(8,2)
 * Impostor: her parmağın 1. izlenimi diğer parmakların 1. izlenimiyle   -> C(P,2)
 *
 * Kullanım: java FvcScores <görüntü_klasörü> <çıktı.csv>
 * Çıktı sütunları: probe,candidate,label,score   (label: 1 = genuine, 0 = impostor)
 */
public class FvcScores {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Kullanım: java FvcScores <klasör> <çıktı.csv>");
            System.exit(2);
        }
        Path dir = Path.of(args[0]);
        Path out = Path.of(args[1]);

        // parmak -> izlenim numaraları (dosya adı: <parmak>_<izlenim>.tif)
        TreeMap<Integer, TreeSet<Integer>> fingers = new TreeMap<>();
        try (var files = Files.list(dir)) {
            for (Path p : (Iterable<Path>) files::iterator) {
                String n = p.getFileName().toString();
                if (!n.toLowerCase(Locale.ROOT).endsWith(".tif")) continue;
                String[] parts = n.substring(0, n.length() - 4).split("_");
                fingers.computeIfAbsent(Integer.parseInt(parts[0]), k -> new TreeSet<>())
                       .add(Integer.parseInt(parts[1]));
            }
        }

        // Şablon çıkarımı (FVC2002 DB1: 500 dpi)
        var opts = new FingerprintImageOptions().dpi(500);
        TreeMap<String, FingerprintTemplate> templates = new TreeMap<>();
        long t0 = System.nanoTime();
        for (var e : fingers.entrySet())
            for (int imp : e.getValue()) {
                String id = e.getKey() + "_" + imp;
                byte[] bytes = Files.readAllBytes(dir.resolve(id + ".tif"));
                templates.put(id, new FingerprintTemplate(new FingerprintImage(bytes, opts)));
            }
        double extractMs = (System.nanoTime() - t0) / 1e6 / templates.size();

        // Çiftler
        List<String[]> pairs = new ArrayList<>();
        for (var e : fingers.entrySet()) {
            List<Integer> imps = new ArrayList<>(e.getValue());
            for (int i = 0; i < imps.size(); i++)
                for (int j = i + 1; j < imps.size(); j++)
                    pairs.add(new String[]{e.getKey() + "_" + imps.get(i), e.getKey() + "_" + imps.get(j), "1"});
        }
        List<Integer> ids = new ArrayList<>(fingers.keySet());
        for (int i = 0; i < ids.size(); i++)
            for (int j = i + 1; j < ids.size(); j++)
                pairs.add(new String[]{ids.get(i) + "_" + fingers.get(ids.get(i)).first(),
                                       ids.get(j) + "_" + fingers.get(ids.get(j)).first(), "0"});

        long t1 = System.nanoTime();
        int gen = 0, imp = 0;
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out))) {
            w.println("probe,candidate,label,score");
            for (String[] pr : pairs) {
                double s = new FingerprintMatcher(templates.get(pr[0])).match(templates.get(pr[1]));
                w.printf(Locale.ROOT, "%s,%s,%s,%.6f%n", pr[0], pr[1], pr[2], s);
                if (pr[2].equals("1")) gen++; else imp++;
            }
        }
        double matchMs = (System.nanoTime() - t1) / 1e6 / pairs.size();

        System.out.printf(Locale.ROOT,
            "%s: %d görüntü, %d genuine, %d impostor | şablon %.1f ms/görüntü, eşleştirme %.3f ms/çift%n",
            dir.getFileName(), templates.size(), gen, imp, extractMs, matchMs);
    }
}
