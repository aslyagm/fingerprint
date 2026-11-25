import com.machinezoo.sourceafis.FingerprintImage;
import com.machinezoo.sourceafis.FingerprintImageOptions;
import com.machinezoo.sourceafis.FingerprintMatcher;
import com.machinezoo.sourceafis.FingerprintTemplate;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/** İki görüntünün SourceAFIS benzerlik skoru. Kullanım: java MatchPair <a.tif> <b.tif> [dpi_a] [dpi_b] */
public class MatchPair {
    static FingerprintTemplate template(String path, double dpi) throws Exception {
        var opts = new FingerprintImageOptions().dpi(dpi);
        return new FingerprintTemplate(new FingerprintImage(Files.readAllBytes(Path.of(path)), opts));
    }

    public static void main(String[] args) throws Exception {
        double da = args.length > 2 ? Double.parseDouble(args[2]) : 500;
        double db = args.length > 3 ? Double.parseDouble(args[3]) : da;
        double s = new FingerprintMatcher(template(args[0], da)).match(template(args[1], db));
        System.out.printf(Locale.ROOT, "%.6f%n", s);
    }
}
