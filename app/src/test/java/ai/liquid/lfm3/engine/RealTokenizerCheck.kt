package ai.liquid.lfm3.engine

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Verifikation gegen die ECHTE LFM3 tokenizer.json (HF LiquidAI/LFM3-1.7B-ONNX).
 *
 * Läuft nur, wenn die Datei vorhanden ist (sonst wird der Test übersprungen):
 *
 *   curl -L -o /tmp/tok_real.json \
 *     "https://huggingface.co/LiquidAI/LFM3-1.7B-ONNX/resolve/main/tokenizer.json"
 *
 * Alternativ Pfad per System-Property überschreiben:
 *   -Dlfm3.real.tokenizer=/pfad/zu/tokenizer.json
 */
class RealTokenizerCheck {

    private fun tok(): LfmTokenizer {
        val f = File(System.getProperty("lfm3.real.tokenizer", "/tmp/tok_real.json"))
        assumeTrue("echte tokenizer.json nicht vorhanden", f.isFile)
        return LfmTokenizer(f.readText())
    }

    @Test
    fun vocabGroesse() {
        assertEquals(124893, tok().vocabSize)
    }

    @Test
    fun echteIds_GPT2Mapping() {
        val tok = tok()
        assertEquals(639, tok.encode(" der")[0])      // Ġder
        assertEquals(35808, tok.encode("Hello")[0])   // Hello
        assertEquals(229, tok.encode(" ")[0])         // Ġ (Leerzeichen!)
        assertEquals(207, tok.encode("\n")[0])        // Ċ (Newline!)
    }

    @Test
    fun roundTrip_deutsch_mitUmlautenUndCJK() {
        val tok = tok()
        val text = "Hallo Welt, wie geht es dir? Heute ist ein schöner Tag. 12345 Grüße, Höhe, Straße. 你好世界 \tTab\nZeile."
        assertEquals(text, tok.decode(tok.encode(text)))
    }

    @Test
    fun encodeMitSpans_konsistentAufEchtemPrompt() {
        val tok = tok()
        val text = "System: Du entscheidest.\nUser: present: button\ntop: 120\nleft: 40\nFrage: Wo ist der Anmelde-Button? <|startoftext|>"
        assertArrayEquals(tok.encode(text), tok.encodeWithSpans(text).first)
    }

    @Test
    fun d1SlotMarker_ohneOverflow() {
        val tok = tok()
        val prompt = ("Du siehst einen Screenshot einer Android-App. Wähle die passende Aktion. " +
            "Optionen: present: Anmelden; top: 120; left: 40; right: 200; bottom: 300. ").repeat(40)
        val t0 = System.currentTimeMillis()
        val ids = tok.encode(prompt)
        val elapsed = System.currentTimeMillis() - t0
        assert(ids.size > 100)
        assert(elapsed < 5_000) { "encode zu langsam: ${elapsed} ms" }
    }

    /**
     * Id-für-Id-Vergleich mit einer exakten HF-Simulation (Python: Split-Regex
     * auf Rohtext, bytes_to_unicode, BPE mit ignore_merges) derselben
     * tokenizer.json. Deckt die drei v1.4.2-Fehlerklassen ab:
     * Rekursion, Bytetabelle, Pre-Tokenizer auf Rohtext.
     */
    @Test
    fun hfGroundTruth_idsWerdenExaktReproduziert() {
        val tok = tok()
        val cases: List<Pair<String, IntArray>> = listOf(
            "Hallo Welt, wie geht es dir?" to intArrayOf(
                108275, 9307, 20, 4065, 16776, 772, 5619, 39),
            "Grüße, Höhe, Straße, Übermorgen, Fahrradstraße" to intArrayOf(
                10205, 641, 6516, 20, 28259, 20, 28287, 20, 12915, 36287, 1157, 20, 14692, 3747, 17102),
            "present: top: 120 left: 40\nbottom: 300" to intArrayOf(
                2156, 34, 2183, 34, 229, 6004, 2690, 34, 229, 1607, 207, 28966, 34, 229, 4590),
            "Zahlen 1234567 und Emoji 😀 ok" to intArrayOf(
                66, 107025, 229, 9792, 25677, 31, 543, 3290, 87, 3465, 53841, 231, 6067),
            "Die Anmelde-Button-Farbe ist #FF0000; URL: https://example.com/pfad?q=1&x=2" to intArrayOf(
                4300, 1091, 18703, 634, 3167, 24226, 4395, 76729, 1667, 2178, 6199, 1372, 24, 35,
                17138, 34, 6496, 1928, 11230, 1949, 8824, 78, 337, 39, 89, 37, 25, 14, 96, 37, 26),
            "Wähle Aktion: present: Anmelden\nKoordinaten: ymin=100 xmin=50 ymax=900 xmax=800" to intArrayOf(
                63, 7386, 298, 334, 7781, 34, 1750, 34, 1091, 18703, 1301, 207, 62612, 40072,
                4191, 34, 382, 1414, 37, 1595, 2552, 1414, 37, 1337, 382, 8207, 37, 10420, 2552, 8207, 37, 7264)
        )
        for ((text, expected) in cases) {
            assertArrayEquals("Text: \"$text\"", expected, tok.encode(text))
            // Round-Trip muss ebenfalls exakt sein
            assertEquals("Round-Trip: \"$text\"", text, tok.decode(tok.encode(text)))
        }
    }
}
