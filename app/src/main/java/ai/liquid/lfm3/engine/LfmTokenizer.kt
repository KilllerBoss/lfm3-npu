package ai.liquid.lfm3.engine

import org.json.JSONObject
import java.util.Locale

/**
 * Byte-Level-BPE-Tokenizer für LFM3 (kompatibel mit HuggingFace tokenizer.json).
 *
 * - Vokabular + Merges aus model.vocab / model.merges
 * - Spezielle Tokens aus dem Top-Level added_tokens-Array (zuverlässigste Quelle),
 *   längste Übereinstimmung zuerst beim Encoding
 * - GPT-2-artige Byte<->Unicode-Tabelle
 * - Inkrementelles UTF-8-Decoding (unvollständige Byte-Folgen werden gepuffert)
 */
class LfmTokenizer(tokenizerJson: String) {

    private val vocab = HashMap<String, Int>(150_000)
    private val invVocab = HashMap<Int, String>(150_000)
    private val merges = HashMap<String, Int>(320_000) // "a b" -> Rang
    private val specialsByLen = ArrayList<Pair<String, Int>>() // sortiert: längste zuerst
    private val specialById = HashMap<Int, String>()
    private val byteToUnicode = arrayOfNulls<String>(256)
    private val unicodeToByte = HashMap<Char, Byte>(256)

    /**
     * Exakter LFM3-Pre-Tokenizer (aus tokenizer.json): Ziffern in 3er-Gruppen,
     * Punktuierung schluckt nachfolgende Zeilenumbrüche (":\n" = ein Pretoken!),
     * ein optionales Nicht-Buchstaben/Nicht-Neue-Zeile-Zeichen vor Wörtern.
     * Wichtig für d1: die Slot-Marker ("present:", "top:" …) müssen exakt so
     * tokenisieren wie das HF-Original.
     */
    private val pretok = Regex(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    )

    init {
        buildByteTables()
        val root = JSONObject(tokenizerJson)

        val model = root.getJSONObject("model")
        val v = model.getJSONObject("vocab")
        val keys = v.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            val id = v.optInt(k, -1)
            if (id >= 0) {
                vocab[k] = id
                invVocab[id] = k
            }
        }
        val mArr = model.optJSONArray("merges")
        if (mArr != null) {
            for (i in 0 until mArr.length()) {
                val item = mArr.get(i)
                val pair: String = when (item) {
                    is String -> item
                    is org.json.JSONArray -> item.optString(0) + " " + item.optString(1)
                    else -> continue
                }
                if (merges.put(pair, i) != null) merges[pair] = i
            }
        }

        val added = root.optJSONArray("added_tokens")
        if (added != null) {
            for (i in 0 until added.length()) {
                val o = added.getJSONObject(i)
                val id = o.optInt("id", -1)
                val content = o.optString("content")
                if (id >= 0 && content.isNotEmpty()) {
                    specialsByLen.add(content to id)
                    specialById[id] = content
                }
            }
        }
        specialsByLen.sortByDescending { it.first.length }
    }

    /** GPT-2-Mapping Bytes -> druckbare Unicode-Zeichen. */
    private fun buildByteTables() {
        val cs = ArrayList<Char>(256)
        var n = 0
        for (b in 0 until 256) {
            val c = when {
                b in 0x21..0x7E -> b.toChar()
                b in 0xA1..0xAC -> b.toChar()
                b in 0xAE..0xFF -> b.toChar()
                else -> {
                    // v1.4.2-Fix: GPT-2 bytes_to_unicode bildet nicht-druckbare
                    // Bytes auf U+0100+n ab (früher fälschlich U+0000+n -> Kollisionen
                    // mit druckbaren Zeichen; Leerzeichen/Newlines wurden beim encode
                    // verworfen und 'Ġ'/'Ċ' beim decode wörtlich ausgegeben).
                    cs.add((0x100 + n).toChar())
                    n++
                    null
                }
            }
            if (c != null) {
                byteToUnicode[b] = c.toString()
                unicodeToByte[c] = b.toByte()
            }
        }
        for (b in 0 until 256) {
            if (byteToUnicode[b] == null) {
                val c = cs.removeAt(0)
                byteToUnicode[b] = c.toString()
                unicodeToByte[c] = b.toByte()
            }
        }
    }

    val vocabSize: Int get() = vocab.size

    fun isSpecial(id: Int): Boolean = specialById.containsKey(id)

    fun specialContent(id: Int): String? = specialById[id]

    /** Vollständiges Encoding inkl. Erkennung spezieller Tokens im Text. */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>(maxOf(16, text.length / 3))
        val seg = StringBuilder()
        var i = 0
        val n = text.length
        fun flush() {
            if (seg.isNotEmpty()) {
                bpeSegment(seg.toString(), out)
                seg.setLength(0)
            }
        }
        while (i < n) {
            var hit: Pair<String, Int>? = null
            if (text[i] == '<') {
                for (sp in specialsByLen) {
                    if (text.startsWith(sp.first, i)) { hit = sp; break }
                }
            }
            if (hit != null) {
                flush()
                out.add(hit.second)
                i += hit.first.length
            } else {
                seg.append(text[i])
                i++
            }
        }
        flush()
        return out.toIntArray()
    }

    private fun bpeSegment(s: String, out: ArrayList<Int>) {
        bpeSegmentWithSpans(s, out, null, 0, null, null)
    }

    /**
     * BPE über ein Segment; wenn [spansOut] gesetzt ist, wird pro Token der
     * Zeichen-Bereich im Originaltext ([segStart] + Offset) mitgeschrieben.
     *
     * v1.4.2-Fix (HF-Exaktheit): der Pre-Tokenizer-Regex läuft – wie in
     * HuggingFace (Split vor ByteLevel) – auf dem ROHTEXT, nicht auf dem
     * byte-gemappten Text. Auf dem gemappten Text verschoben sich die Grenzen
     * ('ü' -> 'Ã'+'¼': '¼' ist \p{N} und trennte das Wort; '\n' -> 'Ċ' ist
     * \p{L} und klebte an Wörter). Danach wird jedes Pretoken byte-gemappt.
     */
    private fun bpeSegmentWithSpans(
        s: String,
        out: ArrayList<Int>,
        spansOut: ArrayList<IntArray>?,
        segStart: Int,
        byteToChar: IntArray?,   // Byte-Index -> Zeichen-Index im Segment
        segChars: Int?           // Zeichenanzahl des Segments
    ) {
        if (s.isEmpty()) return
        var byteOff = 0   // Byte-Offset im Segment (== Länge des gemappten Präfixes)
        var charIdx = 0   // Zeichen-Index im Segment
        for (m in pretok.findAll(s)) {
            // Falls der Regex den Text nicht vollständig abdeckt: Lücke als
            // eigene Pretoken verarbeiten (keine stillen Byte-Verluste).
            if (m.range.first > charIdx) {
                val gap = s.substring(charIdx, m.range.first)
                byteOff = emitPretoken(gap, out, spansOut, segStart, byteToChar, segChars, byteOff)
            }
            byteOff = emitPretoken(m.value, out, spansOut, segStart, byteToChar, segChars, byteOff)
            charIdx = m.range.last + 1
        }
        if (charIdx < s.length) {
            val gap = s.substring(charIdx)
            emitPretoken(gap, out, spansOut, segStart, byteToChar, segChars, byteOff)
        }
    }

    /**
     * Mappt ein Rohtext-Pretoken byte-weise, prüft ignore_merges und führt
     * sonst die BPE aus. Gibt den neuen Byte-Offset zurück.
     */
    private fun emitPretoken(
        raw: String,
        out: ArrayList<Int>,
        spansOut: ArrayList<IntArray>?,
        segStart: Int,
        byteToChar: IntArray?,
        segChars: Int?,
        byteOffIn: Int
    ): Int {
        // WICHTIG: das Pretoken als GANZES nach UTF-8 encodieren. Pro-Zeichen-
        // Encodierung zerstört Surrogat-Paare (Emoji): jedes Halbfahrzeug allein
        // wird von Java durch '?' (0x3F) ersetzt -> falsche Bytes im Model-Input.
        val wordSb = StringBuilder(raw.length)
        for (b in raw.toByteArray(Charsets.UTF_8)) {
            wordSb.append(byteToUnicode[b.toInt() and 0xFF])
        }
        val word = wordSb.toString()
        val mStart = byteOffIn
        val mEnd = byteOffIn + word.length // gemappte Länge == UTF-8-Bytelänge
        val direct = vocab[word]
        if (direct != null) {
            out.add(direct)
            if (spansOut != null) addSpan(spansOut, mStart, mEnd, segStart, byteToChar, segChars, 1)
        } else {
            bpeWordWithSpans(word, out, spansOut, mStart, segStart, byteToChar, segChars)
        }
        return mEnd
    }

    /** Span eines Tokens aus gemappten Byte-Positionen -> Original-Zeichenpositionen. */
    private fun addSpan(
        spansOut: ArrayList<IntArray>, mappedStart: Int, mappedEnd: Int,
        segStart: Int, byteToChar: IntArray?, segChars: Int?, tokenCount: Int
    ) {
        if (tokenCount != 1) return // nur Einzeltoken-Spans hier; BPE-Teile einzeln unten
        val cs = if (byteToChar != null) {
            val b0 = mappedStart; val b1 = mappedEnd
            val c0 = byteToChar.getOrElse(b0) { segChars ?: 0 }
            val c1 = if (b1 >= byteToChar.size) (segChars ?: c0) else byteToChar[b1]
            segStart + c0 to segStart + c1
        } else segStart + mappedStart to segStart + mappedEnd
        spansOut.add(intArrayOf(cs.first, cs.second))
    }

    private fun bpeWordWithSpans(
        word: String,
        out: ArrayList<Int>,
        spansOut: ArrayList<IntArray>?,
        mappedStart: Int,
        segStart: Int,
        byteToChar: IntArray?,
        segChars: Int?
    ) {
        // WICHTIG (v1.4.2): hier NIEMALS an bpeWord() delegieren, wenn spansOut
        // null ist – bpeWord() ruft diese Methode wieder auf => endlose
        // gegenseitige Rekursion => StackOverflowError bei jedem encode() mit
        // einem Wort, das kein Einzel-Vokabular-Token ist (v1.4.0/v1.4.1-Bug).
        // Stattdessen läuft die iterative BPE-Schleife immer hier, die Span-
        // Berechnung wird schlicht übersprungen, wenn spansOut == null.
        if (word.isEmpty()) return
        // 'word' ist bereits byte-gemappt (jedes Zeichen = 1 Byte gemappt) ->
        // Aufteilung in Anfangsteile ist IDENTITÄT. Früher stand hier
        // byteToUnicode[it and 0xFF] (Doppel-Mapping): mit der alten, falschen
        // Bytetabelle zufällig idempotent, seit der Korrektur aber zerstörerisch
        // ('Ł' U+0141 -> 'A' U+0041 usw.) -> Teile nicht im Vokabular -> Bytes
        // wurden still verworfen ("Straße" -> "Strae").
        var parts: List<String> = if (word.length <= 128) {
            word.map { it.toString() }
        } else {
            // Sehr lange Pretokens (z.B. CJK-Ketten) splitten, um O(n²) zu vermeiden
            val res = ArrayList<String>()
            var idx = 0
            while (idx < word.length) {
                val end = minOf(idx + 64, word.length)
                for (c in word.substring(idx, end)) res.add(c.toString())
                idx = end
            }
            res
        }
        while (true) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            var i = 0
            while (i < parts.size - 1) {
                val r = merges[parts[i] + " " + parts[i + 1]]
                if (r != null && r < bestRank) { bestRank = r; bestIdx = i }
                i++
            }
            if (bestIdx < 0) break
            val merged = parts[bestIdx] + parts[bestIdx + 1]
            val np = ArrayList<String>(parts.size)
            for (k in parts.indices) {
                if (k == bestIdx) { np.add(merged) }
                else if (k == bestIdx + 1) { /* übersprungen */ }
                else np.add(parts[k])
            }
            parts = np
        }
        // Teile -> Spans: jeder Teil ist 1 Byte (gemapptes Zeichen) x Länge
        var b = mappedStart
        for (p in parts) {
            val id = vocab[p]
            if (id == null) { b += p.length; continue }
            out.add(id)
            if (spansOut != null) {
                val c0 = if (byteToChar != null) {
                    val cA = byteToChar.getOrElse(b) { segChars ?: 0 }
                    val bEnd = b + p.length
                    val cB = if (bEnd >= byteToChar.size) (segChars ?: cA) else byteToChar[bEnd]
                    segStart + cA to segStart + cB
                } else segStart + b to segStart + b + p.length
                spansOut.add(intArrayOf(c0.first, c0.second))
            }
            b += p.length
        }
    }

    /** Encoding inkl. Zeichen-Spans je Token (für d1-Slot-Marker). */
    fun encodeWithSpans(text: String): Triple<IntArray, IntArray, IntArray> {
        val ids = ArrayList<Int>(text.length / 3)
        val starts = ArrayList<Int>(text.length / 3)
        val ends = ArrayList<Int>(text.length / 3)
        var i = 0
        val n = text.length
        while (i < n) {
            var hit: Pair<String, Int>? = null
            if (text[i] == '<') {
                for (sp in specialsByLen) {
                    if (text.startsWith(sp.first, i)) { hit = sp; break }
                }
            }
            if (hit != null) {
                ids.add(hit.second); starts.add(i); ends.add(i + hit.first.length)
                i += hit.first.length
                continue
            }
            // Segment bis zum nächsten Special sammeln + Byte->Zeichen-Map
            val segStart = i
            val seg = StringBuilder()
            val byteToChar = ArrayList<Int>(seg.length + 16)
            while (i < n) {
                var h: Pair<String, Int>? = null
                if (text[i] == '<') {
                    for (sp in specialsByLen) {
                        if (text.startsWith(sp.first, i)) { h = sp; break }
                    }
                }
                if (h != null) break
                // Pro CODE POINT (nicht pro UTF-16-Char): Surrogat-Paare (Emoji)
                // sind 4 UTF-8-Bytes und dürfen nicht einzeln encodiert werden.
                val cp = Character.codePointAt(text, i)
                val chars = Character.toChars(cp)
                val bytes = String(chars).toByteArray(Charsets.UTF_8)
                repeat(bytes.size) { byteToChar.add(i - segStart) }
                seg.append(chars)
                i += chars.size
            }
            val b2c = byteToChar.toIntArray()
            bpeSegmentWithSpans(seg.toString(), ids, startsEndsToList(starts, ends), segStart, b2c, seg.length)
        }
        return Triple(ids.toIntArray(), starts.toIntArray(), ends.toIntArray())
    }

    private fun startsEndsToList(starts: ArrayList<Int>, ends: ArrayList<Int>): ArrayList<IntArray> {
        // Wrapper, damit bpeSegmentWithSpans beide Listen füllt
        return SpanSink(starts, ends)
    }

    private class SpanSink(val starts: ArrayList<Int>, val ends: ArrayList<Int>) : ArrayList<IntArray>() {
        override fun add(element: IntArray): Boolean {
            starts.add(element[0]); ends.add(element[1]); return true
        }
    }

    /**
     * Entwertet Special-Tokens im Text (Python-plain-Äquivalent): nach dem ersten
     * Zeichen eines Specials wird ein Zero-Width-Space eingefügt, sodass der Text
     * als normaler Text (nicht als Special) tokenisiert wird.
     */
    fun escapeSpecials(textIn: String): String {
        var text = textIn
        var changed = true
        while (changed) {
            changed = false
            val sb = StringBuilder(text.length + 8)
            var i = 0
            while (i < text.length) {
                var hit: String? = null
                if (text[i] == '<') {
                    for (sp in specialsByLen) {
                        if (text.startsWith(sp.first, i)) { hit = sp.first; break }
                    }
                }
                if (hit != null) {
                    sb.append(hit[0]).append('\u200b').append(hit, 1, hit.length)
                    changed = true
                    i += hit.length
                } else {
                    sb.append(text[i]); i++
                }
            }
            text = sb.toString()
        }
        return text
    }

    /** Token-ID -> Token-String (Vokabular-Rückweg). */
    fun tokenForId(id: Int): String? = invVocab[id]

    /** Token-String -> ID (Vokabular-Vorweg; null, wenn unbekannt). */
    fun idFor(token: String): Int? = vocab[token]

    /**
     * Inkrementeller Decoder: wandelt neue Token-IDs in Text um und puffert
     * unvollständige UTF-8-Byte-Folgen (bis zu 3 Nachfolge-Bytes).
     */
    class StreamDecoder(private val tok: LfmTokenizer) {
        private val pending = ByteArray(8)
        private var plen = 0
        private val sb = StringBuilder()

        fun push(id: Int): String {
            sb.setLength(0)
            val sp = tok.specialById[id]
            if (sp != null) return sp // Spezial-Tokens wörtlich ausgeben
            val s = tok.invVocab[id] ?: return ""
            for (c in s) {
                val b = tok.unicodeToByte[c]
                if (b == null) { sb.append(c); continue }
                if (plen >= pending.size) { plen = 0; sb.append('�'); continue }
                pending[plen++] = b
                when (val need = utf8Need()) {
                    0 -> {}                       // unvollständig: warten
                    -1 -> { plen = 0; sb.append('�') } // ungültig: verwerfen
                    else -> {
                        sb.append(String(pending, 0, need, Charsets.UTF_8))
                        plen = 0
                    }
                }
            }
            return sb.toString()
        }

        fun flush(): String {
            if (plen == 0) return ""
            val rest = String(pending, 0, plen, Charsets.UTF_8)
            plen = 0
            return rest
        }

        /** -1 ungültig, 0 unvollständig, sonst benötigte Anzahl Bytes. */
        private fun utf8Need(): Int {
            val c0 = pending[0].toInt() and 0xFF
            val need = when {
                c0 < 0x80 -> 1
                c0 >= 0xF0 -> 4
                c0 >= 0xE0 -> 3
                c0 >= 0xC0 -> 2
                else -> return -1
            }
            if (plen < need) return 0
            for (i in 1 until need) {
                val ci = pending[i].toInt() and 0xFF
                if (ci and 0xC0 != 0x80) return -1
            }
            return need
        }
    }

    /** Einmaliges Dekodieren kompletter ID-Folgen. */
    fun decode(ids: IntArray): String {
        val d = StreamDecoder(this)
        val sb = StringBuilder()
        for (id in ids) sb.append(d.push(id))
        sb.append(d.flush())
        return sb.toString()
    }

    override fun toString(): String =
        String.format(Locale.US, "LfmTokenizer(vocab=%d, merges=%d, specials=%d)",
            vocab.size, merges.size, specialsByLen.size)
}
