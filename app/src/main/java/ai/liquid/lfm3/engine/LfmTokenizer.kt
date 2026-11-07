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

    /** GPT-2-Pre-Tokenizer-Regex (verhindert Merges über Wortgrenzen). */
    private val pretok = Regex(
        "'(?i:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
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
                    cs.add(n.toChar())
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
        if (s.isEmpty()) return
        val mapped = StringBuilder(s.length + 4)
        val bytes = s.toByteArray(Charsets.UTF_8)
        for (b in bytes) mapped.append(byteToUnicode[b.toInt() and 0xFF])
        for (m in pretok.findAll(mapped)) {
            val word = m.value
            val direct = vocab[word]
            if (direct != null) { out.add(direct); continue } // ignore_merges
            bpeWord(word, out)
        }
    }

    private fun bpeWord(word: String, out: ArrayList<Int>) {
        if (word.isEmpty()) return
        var parts: List<String> = if (word.length <= 128) {
            word.chars().mapToObj { byteToUnicode[it and 0xFF]!! }.toList()
        } else {
            // Sehr lange Pretokens (z.B. CJK-Ketten) splitten, um O(n²) zu vermeiden
            val res = ArrayList<String>()
            var idx = 0
            while (idx < word.length) {
                val end = minOf(idx + 64, word.length)
                val chunk = word.substring(idx, end)
                chunk.chars().mapToObj { byteToUnicode[it and 0xFF]!! }.forEach { res.add(it) }
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
        for (p in parts) {
            val id = vocab[p] ?: continue
            out.add(id)
        }
    }

    /** Token-ID -> Token-String (Vokabular-Rückweg). */
    fun tokenForId(id: Int): String? = invVocab[id]

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
