package ai.liquid.lfm3.engine

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM-Tests für LfmTokenizer (kein Android-Code im Tokenizer -> lokal testbar).
 *
 * Regression (v1.4.2): v1.4.0 führte beim encodeWithSpans-Refactor eine endlose
 * gegenseitige Rekursion bpeWord() <-> bpeWordWithSpans() ein. Jeder encode()
 * mit einem Wort, das kein Einzel-Vokabular-Token ist, endete im
 * StackOverflowError ("stack size 4107KB", später "stack size 129MB").
 * Diese Tests erzwingen den BPE-Pfad und würden vor dem Fix sofort überlaufen.
 */
class LfmTokenizerTest {

    private val specialDecide = "<|decide|>"
    private val specialEnd = "<|end|>"

    /** Synthetisches GPT-2-artiges tokenizer.json (Byte-Level-BPE, ignore_merges). */
    private fun newTokenizer(): LfmTokenizer {
        val vocab = JSONObject()
        var id = 0
        fun add(t: String) { vocab.put(t, id++) }

        for (c in 'a'..'z') add(c.toString())
        for (c in 'A'..'Z') add(c.toString())
        add("Ġ")  // GPT-2-Mapping von Leerzeichen (U+0120)
        add(","); add("."); add("!"); add("?")
        add("Ã"); add("¼") // die beiden UTF-8-Bytes von 'ü' im GPT-2-Mapping
        add("1"); add("2"); add("3"); add("0")
        // Ganzwort-Tokens (ignore_merges-Pfad)
        listOf("Ġder", "Ġund", "ĠWelt", "ĠHello", "Ġein").forEach { add(it) }
        // Merge-Ergebnisse müssen im Vokabular liegen, sonst verwirft bpeWord sie
        listOf("GrÃ", "ndungsgesellschaft").forEach { add(it) }
        val idDecide = id++
        val idEnd = id++

        // Merge-Pfade; Rang = Index. Achtung: der Pre-Tokenizer läuft auf dem
        // gemappten Text und trennt an '¼' (\p{N}) – deshalb trennen sich die
        // Pfade "GrÃ" | "¼" | "ndungs…" (echtes GPT-2-Verhalten bei Umlauten).
        val merges = JSONArray()
        listOf(
            "G r", "Gr Ã",                                 // -> GrÃ (1. Pretoken)
            "n d", "nd u", "ndu n", "ndun g", "ndung s",   // -> ndungs…
            "ndungs g", "ndungsg e", "ndungsge s", "ndungsges e",
            "ndungsgese l", "ndungsgesel l", "ndungsgesell s", "ndungsgesells c",
            "ndungsgesellsc h", "ndungsgesellsch a", "ndungsgesellscha f",
            "ndungsgesellschaf t",                         // -> ndungsgesellschaft
            "Ġ d", "Ġd e", "Ġde r",                       // -> Ġder
            "Ġ u", "Ġu n", "Ġun d",                       // -> Ġund
            "Ġ W", "ĠW e", "ĠWe l", "ĠWel t",             // -> ĠWelt
            "Ġ H", "ĠH e", "ĠHe l", "ĠHel l", "ĠHell o",  // -> ĠHello
            "Ġ e", "Ġe i", "Ġei n"                        // -> Ġein
        ).forEach { merges.put(it) }

        val added = JSONArray()
        fun special(content: String, sid: Int) {
            val o = JSONObject()
            o.put("id", sid)
            o.put("content", content)
            added.put(o)
        }
        special(specialDecide, idDecide)
        special(specialEnd, idEnd)

        val model = JSONObject().put("vocab", vocab).put("merges", merges)
        val root = JSONObject().put("model", model).put("added_tokens", added)
        return LfmTokenizer(root.toString())
    }

    @Test
    fun directVocabHit_wirdAlsEinTokenAusgegeben() {
        val tok = newTokenizer()
        val ids = tok.encode(" der")
        assertEquals(1, ids.size)
        assertEquals(tok.idFor("Ġder"), ids[0])
    }

    @Test
    fun unbekanntesWort_fälltAufEinzelbytesZurück() {
        val tok = newTokenizer()
        val ids = tok.encode("xyzq")
        assertEquals(listOf("x", "y", "z", "q").map { tok.idFor(it) }, ids.toList())
    }

    @Test
    fun bpePfad_mergedMehrbyteWortUndTerminiert() {
        // Vor dem Fix: StackOverflowError genau hier (Wort nicht im Vokabular).
        val tok = newTokenizer()
        val ids = tok.encode("Gründungsgesellschaft")
        // Erwartung (echtes GPT-2-Verhalten): der Pre-Tokenizer trennt den
        // gemappten Text "GrÃ¼ndungsgesellschaft" an '¼' (\p{N}):
        //   "GrÃ" (BPE) | "¼" (Einzeltoken) | "ndungsgesellschaft" (BPE-Kette)
        assertEquals(
            listOf(tok.idFor("GrÃ"), tok.idFor("¼"), tok.idFor("ndungsgesellschaft")),
            ids.toList()
        )
    }

    @Test
    fun encode_und_encodeWithSpans_liefernIdentischeIds() {
        val tok = newTokenizer()
        val texts = listOf(
            "Hello der Welt",
            "Gründungsgesellschaft und so weiter, mit Zahlen 12345!",
            "Ein langer Satz: der Plan ist fertig.\nNeue Zeile\tTab.",
            "a${specialDecide}b Hallo${specialEnd}Test",
            "Frage: present: top: left: right:\nAntwort mit Ümläuten und ß?",
            ""
        )
        for (t in texts) {
            assertArrayEquals("Text: \"$t\"", tok.encode(t), tok.encodeWithSpans(t).first)
        }
    }

    @Test
    fun decode_roundTrip() {
        val tok = newTokenizer()
        val text = "Hello der Welt, Gründungsgesellschaft! 123"
        assertEquals(text, tok.decode(tok.encode(text)))
    }

    @Test
    fun decode_sonderTokenWörtlich() {
        val tok = newTokenizer()
        val text = "a${specialDecide}b"
        assertEquals(text, tok.decode(tok.encode(text)))
    }

    @Test
    fun spans_sindMonotonUndInGrenzen() {
        val tok = newTokenizer()
        val text = "Gründungs Welt,Hallo"
        val (ids, starts, ends) = tok.encodeWithSpans(text)
        assertEquals(ids.size, starts.size)
        assertEquals(ids.size, ends.size)
        var prevStart = -1
        for (i in ids.indices) {
            assertTrue("start $i < 0", starts[i] >= 0)
            assertTrue("end $i > len", ends[i] <= text.length)
            assertTrue("start $i sinkt", starts[i] >= prevStart)
            assertTrue("start>=end bei $i", starts[i] < ends[i])
            prevStart = starts[i]
        }
    }

    @Test
    fun stress_langeWoerterUndTexte_terminierenSchnell() {
        val tok = newTokenizer()
        val longWord = "x".repeat(4000)              // > 128 -> Chunked-Pfad
        val longText = ("Hallo Welt, Gründungsgesellschaft 42 $specialDecide ").repeat(300)
        val t0 = System.currentTimeMillis()
        val a = tok.encode(longWord)
        val b = tok.encode(longText)
        val c = tok.encodeWithSpans(longText)
        val elapsed = System.currentTimeMillis() - t0
        assertEquals(a.toList(), tok.encodeWithSpans(longWord).first.toList())
        assertArrayEquals(b, c.first)
        assertTrue("Tokenizer zu langsam: $elapsed ms", elapsed < 10_000)
    }
}
