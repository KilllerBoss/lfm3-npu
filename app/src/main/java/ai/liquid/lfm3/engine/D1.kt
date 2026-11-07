package ai.liquid.lfm3.engine

import org.json.JSONArray
import org.json.JSONObject
import java.util.Locale
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * System-One-Protokoll für d1-Entscheidungsmodelle (LiquidAI d1-2B-ONNX).
 *
 * Port der Referenz-Implementierung (prompt.py / box.py / grid.py / api.py /
 * calibration.py) auf Kotlin. Ein Decision läuft in EINEM Forward-Pass:
 * State + Bild + bis zu 16 Fragen werden in eine Zeile gepackt
 * (Set-Attention über question_ids), und die Antwort wird aus den Logits an
 * den Antwort-Slots gelesen (keine Token-Generierung).
 *
 * Checkpoint-Konstanten aus config.json von LiquidAI/d1-2B-ONNX:
 * d1_codes=2, d1_attention="set", d1_numbers=true, d1_set_size=16,
 * d1_box_edges="median", d1_prompt=1, d1_vision_max_pixels=1638400.
 */
object D1 {

    // ------------------------------------------------------------ Konstanten

    const val BOS = "<|startoftext|>"
    const val IM_START = "<|im_start|>"
    const val IM_END = "<|im_end|>"
    const val LEAD = "<think></think>"          // Chat-Template, enable_thinking=false
    const val VISION_START = "<|vision_start|>"
    const val IMAGE_TOK = "<|image|>"
    const val VISION_END = "<|vision_end|>"
    const val QUESTION_HEADER = "QUESTION:\n"

    const val D1_CODES = 2
    const val D1_SET_SIZE = 16                  // max. Fragen pro Forward-Pass
    const val D1_BOX_EDGES = "median"
    const val D1_PROMPT_VERSION = 1
    const val BOX_INSTANCES = 16                // boxes-Blöcke ohne "max" (Prompt 1)
    const val BOX_MAX_INSTANCES = 64

    val EDGES = arrayOf("top", "bottom", "left", "right")

    private val YES_FORMS = arrayOf("yes", "Yes", "YES")
    private val NO_FORMS = arrayOf("no", "No", "NO")

    /** Python "%g"-Formatierung (6 signifikante Stellen, ganze Zahlen pur). */
    fun g(v: Double): String {
        if (v.isNaN() || v.isInfinite()) return v.toString()
        if (v == floor(v) && abs(v) < 1e15) return v.toLong().toString()
        var s = String.format(Locale.US, "%.6g", v)
        if (s.contains('e')) {
            s = java.math.BigDecimal(v).stripTrailingZeros().toPlainString()
        }
        return s
    }

    private fun bisectLeft(arr: DoubleArray, v: Double): Int {
        var lo = 0
        var hi = arr.size
        while (lo < hi) {
            val mid = (lo + hi) / 2
            if (arr[mid] < v) lo = mid + 1 else hi = mid
        }
        return lo
    }

    // ---------------------------------------------------------------- Grid

    class Grid(val low: Double, val high: Double, val size: Int) {
        val step: Double get() = (high - low) / (size - 1)
        val values: DoubleArray by lazy {
            DoubleArray(size) { i -> if (i == size - 1) high else low + i * step }
        }
    }

    val EDGE_CODE_GRID = Grid(0.0, 1.0, 128)
    val THOUSANDTHS = Grid(0.0, 0.999, 1000)

    // ------------------------------------------------------------ Fragen

    sealed class Q {
        abstract val instructions: String
        abstract val type: String
        /** Wie viele Logit-Slots diese Frage in der Zeile belegt. */
        abstract fun slotCount(): Int
    }

    class Noul(override val instructions: String,
               val trueText: String?, val falseText: String?) : Q() {
        override val type = "noul"
        override fun slotCount() = 1
    }

    class Choice(override val instructions: String,
                 val criteria: List<Pair<String, String?>>,   // (Name, Beschreibung)
                 val multiple: Boolean) : Q() {
        override val type = "choice"
        override fun slotCount() = if (multiple) criteria.size else 1
    }

    class Score(override val instructions: String,
                val criteria: List<String>) : Q() {
        override val type = "score"
        override fun slotCount() = 1
    }

    /** Zahl schätzen (mit range) – Distribution über gleichmäßig verteilte Werte. */
    class Estimate(override val instructions: String,
                   val low: Double, val high: Double,
                   val step: Double?, val unit: String?) : Q() {
        override val type = "number"
        val grid: Grid = Grid(low, high,
            if (step == null) 128 else min(128, ((high - low) / step).roundToInt() + 1))
        override fun slotCount() = 1
    }

    class Box(override val instructions: String) : Q() {
        override val type = "box"
        override fun slotCount() = 5   // presence + 4 Kanten
    }

    class Boxes(override val instructions: String, val max: Int?) : Q() {
        override val type = "boxes"
        val blocks: Int get() = max ?: BOX_INSTANCES
        override fun slotCount() = blocks * 5
    }

    data class NamedQ(val name: String, val q: Q)

    /** Liest Fragen aus {"name": {type, instructions, criteria…}} und validiert sie. */
    fun parseQuestions(questions: JSONObject): List<NamedQ> {
        if (questions.length() == 0) throw IllegalArgumentException("questions ist leer")
        if (questions.length() > D1_SET_SIZE) throw IllegalArgumentException(
            "Maximal $D1_SET_SIZE Fragen pro Aufruf (d1_set_size)")
        val out = ArrayList<NamedQ>()
        for (name in questions.keys()) {
            val o = questions.optJSONObject(name)
                ?: throw IllegalArgumentException("Frage '$name': erwartet ein Objekt mit \"type\" und \"instructions\"")
            val kind = o.optString("type", "choice")
            val said = o.optString("instructions", "").trim()
            if (said.isEmpty()) throw IllegalArgumentException("Frage '$name': \"instructions\" fehlt")
            when (kind) {
                "noul" -> out.add(NamedQ(name, Noul(said,
                    o.optJSONObject("criteria")?.optString("true"),
                    o.optJSONObject("criteria")?.optString("false"))))
                "choice" -> {
                    val c = o.optJSONObject("criteria")
                        ?: throw IllegalArgumentException("Frage '$name': choice braucht \"criteria\" als Objekt {option: beschreibung}")
                    val list = ArrayList<Pair<String, String?>>()
                    for (k in c.keys()) list.add(k to (if (c.isNull(k)) null else c.optString(k)))
                    if (list.isEmpty()) throw IllegalArgumentException("Frage '$name': choice ohne Optionen")
                    out.add(NamedQ(name, Choice(said, list, o.optBoolean("multiple", false))))
                }
                "score" -> {
                    val arr = o.optJSONArray("criteria")
                        ?: throw IllegalArgumentException("Frage '$name': score braucht \"criteria\" als Liste von 2-10 Stufen")
                    val list = ArrayList<String>()
                    for (i in 0 until arr.length()) list.add(arr.optString(i))
                    if (list.size !in 2..10) throw IllegalArgumentException(
                        "Frage '$name': score braucht 2-10 Stufen, hat ${list.size}")
                    out.add(NamedQ(name, Score(said, list)))
                }
                "number" -> {
                    val r = o.optJSONArray("range")
                        ?: throw IllegalArgumentException("Frage '$name': number braucht \"range\": [low, high] (Zahlen aus Text extrahieren wird nicht unterstützt)")
                    if (r.length() != 2) throw IllegalArgumentException("Frage '$name': range braucht genau 2 Werte [low, high]")
                    val low = r.optDouble(0)
                    val high = r.optDouble(1)
                    if (!(low < high)) throw IllegalArgumentException("Frage '$name': range muss von low nach high laufen")
                    val st = if (o.has("step") && !o.isNull("step")) o.optDouble("step") else null
                    if (st != null && st <= 0) throw IllegalArgumentException("Frage '$name': step muss > 0 sein")
                    val unit = o.optString("unit", "")
                    out.add(NamedQ(name, Estimate(said, low, high, st, unit.ifEmpty { null })))
                }
                "text", "date" -> throw IllegalArgumentException(
                    "Frage '$name': Typ '$kind' (Extraktion aus dem State) wird von dieser Engine noch nicht unterstützt; nutze choice/score/number/box/boxes")
                "box" -> out.add(NamedQ(name, Box(said)))
                "boxes" -> {
                    val m = if (o.has("max") && !o.isNull("max")) o.optInt("max", -1) else null
                    if (m != null && (m < 1 || m > BOX_MAX_INSTANCES)) throw IllegalArgumentException(
                        "Frage '$name': max muss zwischen 1 und $BOX_MAX_INSTANCES liegen")
                    out.add(NamedQ(name, Boxes(said, m)))
                }
                else -> throw IllegalArgumentException(
                    "Frage '$name': type ist einer von noul, choice, score, number (mit range), box, boxes – nicht '$kind'")
            }
        }
        return out
    }

    // ------------------------------------------------------- Kalibrierung

    class Calibration(
        val temperatures: Map<String, Double>,
        val default: Double,
        val tables: Map<String, Array<DoubleArray>>
    ) {
        fun temperature(bucket: String): Double = temperatures[bucket] ?: default

        fun apply(bucket: String, scores: DoubleArray): DoubleArray {
            val t = temperature(bucket)
            if (t == 1.0) return scores
            return DoubleArray(scores.size) { scores[it] / t }
        }

        private fun through(value: Double, knots: Array<DoubleArray>?): Double {
            if (knots == null || knots.isEmpty()) return value
            var idx = knots.indexOfFirst { it[0] >= value }
            if (idx < 0) return knots.last()[1]
            if (idx == 0) return knots[0][1]
            val aRaw = knots[idx - 1][0]; val aCal = knots[idx - 1][1]
            val bRaw = knots[idx][0]; val bCal = knots[idx][1]
            if (bRaw == aRaw) return bCal
            return aCal + (value - aRaw) * (bCal - aCal) / (bRaw - aRaw)
        }

        /** Kalibriert einen Rohwert über die Tabelle (z. B. "found:box"). */
        fun calibrate(table: String, value: Double): Double = through(value, tables[table])

        /** Quantil-Level, an dem ein Anteil gelesen wird (estimate-Tabelle rückwärts). */
        fun level(share: Double): Double {
            val rev = tables["estimate"]?.map { doubleArrayOf(it[1], it[0]) }?.toTypedArray()
            return through(share, rev)
        }

        companion object {
            fun parse(json: String?): Calibration {
                if (json == null) return Calibration(emptyMap(), 1.0, emptyMap())
                return try {
                    val d = JSONObject(json)
                    val temps = HashMap<String, Double>()
                    d.optJSONObject("temperatures")?.let { t ->
                        for (k in t.keys()) temps[k] = t.optDouble(k)
                    }
                    val tables = HashMap<String, Array<DoubleArray>>()
                    d.optJSONObject("tables")?.let { ts ->
                        for (k in ts.keys()) {
                            val arr = ts.optJSONArray(k) ?: continue
                            val rows = Array(arr.length()) { i ->
                                val row = arr.optJSONArray(i)!!
                                doubleArrayOf(row.optDouble(0), row.optDouble(1))
                            }
                            tables[k] = rows
                        }
                    }
                    Calibration(temps, d.optDouble("default", 1.0), tables)
                } catch (_: Throwable) {
                    Calibration(emptyMap(), 1.0, emptyMap())
                }
            }
        }
    }

    // --------------------------------------------------------- Frage-Texte

    /** question_block aus prompt.py – exakte Texte, Prompt-Version 1. */
    fun questionBlock(tok: LfmTokenizer, q: Q): String {
        val said = tok.escapeSpecials(q.instructions)
        return when (q) {
            is Box -> {
                val numbers = edgeNumbersAvailable(tok)
                val legend = if (numbers) EDGE_LEGEND_NUMBERS else codeLegend(EDGE_CODE_GRID)
                val one = if (numbers) "one number" else "one code"
                "$said\n\nPresence is yes or no. $legend Reply with yes or no at present, " +
                    "then $one at top, bottom, left and right."
            }
            is Boxes -> {
                val numbers = edgeNumbersAvailable(tok)
                val legend = if (numbers) EDGE_LEGEND_NUMBERS else codeLegend(EDGE_CODE_GRID)
                val most = if (D1_PROMPT_VERSION == 1) "16 at most, " else ""
                "$said\n\nEvery one of them, ${most}in reading order (left edge, then top). " +
                    "Present k is yes when there are at least k. $legend Reply with yes or no at each " +
                    "present, then its top, bottom, left and right."
            }
            is Estimate -> {
                val unit = q.unit?.let { " $it" } ?: ""
                if (numberIdsAvailable(tok, q.grid, 1)) {
                    "$said\n\nEstimate from ${g(q.low)} to ${g(q.high)}$unit, in steps of " +
                        "${g(q.grid.step)}. Reply with the number only."
                } else {
                    val codes = estimateCodes(q.grid)
                    "$said\n\nEstimate from ${g(q.low)} to ${g(q.high)}$unit on " +
                        "${q.grid.size} equally spaced values, step ${g(q.grid.step)}: " +
                        "${codes[0]} is ${g(q.low)}, ${codes[codes.size - 1]} is ${g(q.high)}. Reply with the code only."
                }
            }
            is Choice -> {
                if (q.multiple) {
                    val lines = q.criteria.joinToString("\n") { (name, desc) ->
                        "${jsonQuote(name)}: ${desc ?: name.replace('_', ' ')}"
                    }
                    "$said\n\nOptions:\n$lines\n\nReply with yes or no for each option."
                } else {
                    val codes = optionCodes(q.criteria.map { it.first })
                    val lines = StringBuilder()
                    for (i in q.criteria.indices) {
                        if (i > 0) lines.append('\n')
                        val (label, desc) = q.criteria[i]
                        val text = desc ?: label.replace('_', ' ')
                        lines.append(codes[i]).append(' ').append(text)
                    }
                    "$said\n\nOptions:\n$lines\n\nReply with the option code only."
                }
            }
            is Noul -> {
                val extra = if (q.trueText != null || q.falseText != null) {
                    "\nYes: ${q.trueText ?: ""}\nNo: ${q.falseText ?: ""}"
                } else ""
                "$said$extra\n\nReply with yes or no only."
            }
            is Score -> {
                val legend = q.criteria.indices.joinToString("\n") { i -> "$i ${q.criteria[i]}" }
                "$said\n\n$legend\n\nReply with a single digit 0-${q.criteria.size - 1} only."
            }
        }
    }

    private const val EDGE_LEGEND_NUMBERS =
        "Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom " +
        "from its top edge, left and right from its left edge."

    private fun codeLegend(grid: Grid): String {
        val codes = optionCodes((0 until grid.size).map { it.toString() })
        return "Each edge is a code on a ${grid.size}-value scale: ${codes[0]} is 0 (top or left), " +
            "${codes[codes.size - 1]} is 1 (bottom or right). Values are equally spaced."
    }

    private fun jsonQuote(s: String): String = JSONObject.quote(s)

    /** option_codes, Scheme 2 (d1_codes=2): A..Z, dann AA, AB, … */
    fun optionCodes(labels: List<String>): List<String> {
        val labs = labels.map { it.trim() }
        if (labs.isNotEmpty() && labs.all { it.length == 1 && it[0].isLetter() }) return labs
        if (labs.size <= 26) return (0 until labs.size).map { ('A' + it).toString() }
        val out = ArrayList<String>(labs.size)
        for (i in 0 until labs.size) {
            out.add(if (i < 26) ('A' + i).toString() else "${'A' + (i - 26) / 26}${'A' + (i - 26) % 26}")
        }
        return out
    }

    private fun gridQuestion(grid: Grid): Q =
        if (grid.size <= 10) Score("", grid.values.map { g(it) })
        else Choice("", grid.values.mapIndexed { i, v -> i.toString() to g(v) }, multiple = false)

    private fun estimateCodes(grid: Grid): List<String> =
        if (grid.size <= 10) (0 until grid.size).map { it.toString() }
        else optionCodes((0 until grid.size).map { it.toString() })

    // --------------------------------------------------- Readout-Token-IDs

    private fun singleId(tok: LfmTokenizer, s: String): Int? {
        val e = tok.encode(s)
        return if (e.size == 1) e[0] else null
    }

    /** number_ids: für jeden Grid-Wert die (einfachen) Token-Formen; null wenn nicht als Zahl lesbar. */
    private fun numberIds(tok: LfmTokenizer, grid: Grid, scale: Int): Array<IntArray>? {
        val groups = ArrayList<IntArray>(grid.size)
        for (v in grid.values) {
            val n = Math.round(v * scale)
            if (abs(n - v * scale) > 1e-6) return null
            val plain = singleId(tok, n.toString()) ?: return null
            val spaced = singleId(tok, " $n")
            groups.add(if (spaced != null && spaced != plain) intArrayOf(plain, spaced) else intArrayOf(plain))
        }
        return groups.toTypedArray()
    }

    fun numberIdsAvailable(tok: LfmTokenizer, grid: Grid, scale: Int): Boolean =
        numberIds(tok, grid, scale) != null

    fun edgeNumbersAvailable(tok: LfmTokenizer): Boolean = numberIdsAvailable(tok, THOUSANDTHS, 1000)

    /** Gruppen von Token-IDs pro Option (max-gepoolt): die Antwort ist ein Softmax genau über diese. */
    fun readoutGroups(tok: LfmTokenizer, q: Q): Array<IntArray> = when (q) {
        is Noul -> {
            val yes = YES_FORMS.mapNotNull { singleId(tok, it) }
            val no = NO_FORMS.mapNotNull { singleId(tok, it) }
            if (yes.isEmpty() || no.isEmpty()) throw IllegalStateException("Tokenizer hat keine yes/no-Einzeltokens")
            arrayOf(yes.toIntArray(), no.toIntArray())
        }
        is Score -> q.criteria.indices.map { i ->
            intArrayOf(singleId(tok, i.toString())
                ?: throw IllegalStateException("Score braucht Einzeltoken-Ziffern"))
        }.toTypedArray()
        is Choice -> {
            if (q.multiple) throw IllegalStateException("multiple choice liest yes/no je Option")
            val codes = optionCodes(q.criteria.map { it.first })
            codes.map { code ->
                val main = singleId(tok, code) ?: throw IllegalStateException("Kein Einzeltoken für Code $code")
                val spaced = singleId(tok, " $code")
                if (spaced != null && spaced != main) intArrayOf(main, spaced) else intArrayOf(main)
            }.toTypedArray()
        }
        is Estimate -> {
            numberIds(tok, q.grid, 1)
                ?: gridReadoutGroups(tok, gridQuestion(q.grid), q.grid.size)
        }
        is Box -> throw IllegalStateException("Box-Slots einzeln lesen (presence/edges)")
        is Boxes -> throw IllegalStateException("Boxes-Slots einzeln lesen")
    }

    /** Readout-Gruppen für ein Grid über den Code-Weg (Fallback ohne Zahlentokens). */
    fun gridReadoutGroups(tok: LfmTokenizer, slotQ: Q, size: Int): Array<IntArray> {
        val groups = readoutGroups(tok, slotQ)
        require(groups.size == size) { "Grid-Readout: ${groups.size} Gruppen != $size Werte" }
        return groups
    }

    // ------------------------------------------------------ Prompt-Aufbau

    class SlotLayout(val suffixText: String, val slotOffsets: IntArray) // Offsets relativ zum letzten Token

    /**
     * marked_suffix: Frage-Body + Assistant-Header + Lead + Marker ("present:",
     * "top:" …); liefert den Text und die Token-Offsets der Slots (negativ vom
     * letzten Token). Single-Slot-Fragen enden direkt nach dem Lead.
     */
    fun suffixLayout(tok: LfmTokenizer, q: Q): SlotLayout {
        val body = questionBlock(tok, q)
        val names: List<String> = when (q) {
            is Boxes -> (1..q.blocks).flatMap { k -> listOf("present", *EDGES).map { n -> "$n $k" } }
            is Box -> listOf("present", *EDGES)
            is Choice -> if (q.multiple) q.criteria.map { jsonQuote(it.first) } else emptyList()
            else -> emptyList()
        }
        if (names.isEmpty()) {
            return SlotLayout("${tok.escapeSpecials(body)}$IM_END\n$IM_START" +
                "assistant\n$LEAD", intArrayOf(0))
        }
        var text = "${tok.escapeSpecials(body)}$IM_END\n$IM_START" + "assistant\n$LEAD"
        val ends = ArrayList<Int>(names.size)
        for ((i, name) in names.withIndex()) {
            if (i > 0) text += "\n"
            text += tok.escapeSpecials(name) + ":"
            ends.add(text.length)
        }
        // Nachgestelltes "\n", wenn der letzte Colon-Token den Umbruch schluckt (Boxes, Prompt 1)
        if (q is Boxes && (q.blocks < BOX_INSTANCES || D1_PROMPT_VERSION > 1)) {
            val probe = tok.encode(text + "\n")
            val lastTok = probe.lastOrNull()?.let { tok.decode(intArrayOf(it)) } ?: ""
            if (lastTok.trimEnd().endsWith(":")) text += "\n"
        }
        val (ids, starts, endsSpans) = tok.encodeWithSpans(text)
        val slots = IntArray(ends.size)
        for ((i, chars) in ends.withIndex()) {
            var found = -1
            for (t in ids.indices) {
                if (starts[t] <= chars - 1 && chars - 1 < endsSpans[t]) { found = t; break }
            }
            if (found < 0) throw IllegalStateException("Slot-Marker '${names[i]}' hat keine exakte Colon-Token-Position")
            val tokText = tok.decode(intArrayOf(ids[found]))
            val after = text.substring(min(chars, text.length), endsSpans[found])
            if (after.trim().isNotEmpty() || !tokText.trimEnd().endsWith(":")) {
                throw IllegalStateException("Slot-Marker-Token von '${names[i]}' endet nicht mit ':'")
            }
            slots[i] = found - ids.size + 1
        }
        return SlotLayout(text, slots)
    }

    /** Ein Forward-Pass: eine gepackte Zeile mit State + bis zu 16 Fragen. */
    class D1Pass(
        val inputIds: IntArray,
        val questionIds: IntArray,      // -1 State, 1..N Fragen
        val slotPositions: IntArray,    // flat, Frage-für-Frage
        val questionIndices: IntArray,  // globale Indizes der Fragen in dieser Zeile
        val slotCount: Int
    )

    /**
     * Baut die gepackten Zeilen: State + Bild-Markup + Fragen im Set-Layout
     * (pack_set-Port). Mehr als 16 Fragen => mehrere Zeilen über denselben State.
     */
    fun buildPasses(tok: LfmTokenizer, state: String?, imageTokenCounts: IntArray,
                    questions: List<NamedQ>): List<D1Pass> {
        val markup = imageTokenCounts.joinToString("") { n ->
            VISION_START + IMAGE_TOK.repeat(n) + VISION_END
        }
        val header = if (state != null) QUESTION_HEADER else ""
        val body = if (state != null) tok.escapeSpecials("$state\n\n") + header else ""
        val prefix = BOS + IM_START + "user\n" + markup + body

        val passes = ArrayList<D1Pass>()
        var qi = 0
        while (qi < questions.size) {
            val group = questions.subList(qi, min(questions.size, qi + D1_SET_SIZE))
            val layouts = group.map { suffixLayout(tok, it.q) }
            val first = layouts[0]
            var rowIds = tok.encode(prefix + first.suffixText)
            val stateLen = rowIds.size - tok.encode(first.suffixText).size
            val qids = ArrayList<Int>(rowIds.size)
            val slotPos = ArrayList<Int>(32)
            repeat(stateLen) { qids.add(-1) }
            repeat(rowIds.size - stateLen) { qids.add(1) }
            for (off in first.slotOffsets) slotPos.add(rowIds.size - 1 + off)

            var qk = 1
            for (j in 1 until group.size) {
                qk += 1
                val own = tok.encode(header + layouts[j].suffixText)
                val oldSize = rowIds.size
                rowIds = rowIds.copyOf(oldSize + own.size)
                System.arraycopy(own, 0, rowIds, oldSize, own.size)
                repeat(own.size) { qids.add(qk) }
                for (off in layouts[j].slotOffsets) slotPos.add(oldSize + own.size - 1 + off)
            }
            val indices = IntArray(group.size) { k -> qi + k }
            var slots = 0
            for (nq in group) slots += nq.q.slotCount()
            passes.add(D1Pass(rowIds, qids.toIntArray(), slotPos.toIntArray(), indices, slots))
            qi += group.size
        }
        return passes
    }

    // ------------------------------------------------------- Readout/Softmax

    fun softmax(scores: DoubleArray): DoubleArray {
        var m = Double.NEGATIVE_INFINITY
        for (s in scores) if (s > m) m = s
        val exps = DoubleArray(scores.size) { exp(scores[it] - m) }
        val total = exps.sum()
        return DoubleArray(exps.size) { exps[it] / total }
    }

    /** max-gepoolte Gruppen-Scores -> optional kalibriert -> Softmax. */
    private fun groupProbs(logits: FloatArray, groups: Array<IntArray>, calibration: Calibration?,
                           bucket: String?, tempered: Boolean): DoubleArray {
        val scores = DoubleArray(groups.size)
        for ((i, grp) in groups.withIndex()) {
            var best = Double.NEGATIVE_INFINITY
            for (id in grp) if (id < logits.size) {
                val v = logits[id].toDouble()
                if (v > best) best = v
            }
            scores[i] = best
        }
        val used = if (tempered && calibration != null && bucket != null) calibration.apply(bucket, scores) else scores
        return softmax(used)
    }

    // ---------------------------------------------------------- Box-Mathe

    private fun ordered(a: Double, b: Double, grid: Grid): Pair<Double, Double> {
        if (a < b) return a to b
        val half = grid.step / 2
        val middle = min(1 - half, maxOf(half, (a + b) / 2))
        return (middle - half) to (middle + half)
    }

    private fun rectangle(edges: DoubleArray, grid: Grid): JSONObject {
        val (top, bottom) = ordered(edges[0], edges[1], grid)
        val (left, right) = ordered(edges[2], edges[3], grid)
        return JSONObject().put("top", top).put("bottom", bottom)
            .put("left", left).put("right", right)
    }

    private fun area(b: JSONObject): Double =
        (b.getDouble("bottom") - b.getDouble("top")) * (b.getDouble("right") - b.getDouble("left"))

    private fun iou(a: JSONObject, b: JSONObject): Double {
        val overlapW = maxOf(0.0, min(a.getDouble("bottom"), b.getDouble("bottom")) - maxOf(a.getDouble("top"), b.getDouble("top")))
        val overlapH = maxOf(0.0, min(a.getDouble("right"), b.getDouble("right")) - maxOf(a.getDouble("left"), b.getDouble("left")))
        val overlap = overlapW * overlapH
        val union = area(a) + area(b) - overlap
        return if (union <= 0) 0.0 else overlap / union
    }

    /** Halton-Sequenz (512 Punkte, Basen 2,3,5,7) – wie box.samples(). */
    private val halton: Array<DoubleArray> by lazy {
        fun radicalInverse(i: Int, base: Int): Double {
            var v = i; var scale = 1.0; var value = 0.0
            while (v > 0) {
                val digit = v % base
                v /= base
                scale /= base
                value += digit * scale
            }
            return value
        }
        val bases = intArrayOf(2, 3, 5, 7)
        Array(512) { i -> DoubleArray(4) { j -> radicalInverse(i + 1, bases[j]) } }
    }

    /** Anteil der aus den Kantendistributionen gezogenen Boxen mit IoU >= 0.5. */
    private fun boxConfidence(edgeProbs: List<DoubleArray>, box: JSONObject): Double {
        val cdfs = edgeProbs.map { row ->
            val total = row.sum()
            val c = DoubleArray(row.size)
            var acc = 0.0
            for (i in row.indices) { acc += row[i] / total; c[i] = acc }
            c
        }
        val grid = if (edgeProbs[0].size == THOUSANDTHS.size) THOUSANDTHS else EDGE_CODE_GRID
        val values = grid.values
        var hits = 0
        for (point in halton) {
            val e = DoubleArray(4) { k ->
                val idx = bisectLeft(cdfs[k], point[k])
                values[min(grid.size - 1, idx)]
            }
            if (iou(rectangle(e, grid), box) >= 0.5) hits++
        }
        return hits / 512.0
    }

    /** Die vier Kantendistributionen -> Box an den Medianen (d1_box_edges="median"). */
    private fun medians(edges: List<DoubleArray>): JSONObject {
        val grid = if (edges[0].size == THOUSANDTHS.size) THOUSANDTHS else EDGE_CODE_GRID
        val vals = DoubleArray(4) { k ->
            val total = edges[k].sum().coerceAtLeast(1e-12)
            var mass = 0.0
            var med = grid.values[grid.size - 1]
            for (i in edges[k].indices) {
                mass += edges[k][i] / total
                if (mass >= 0.5) { med = grid.values[i]; break }
            }
            med
        }
        return rectangle(vals, grid)
    }

    private fun edgeDistribution(logits: FloatArray, tok: LfmTokenizer): DoubleArray {
        val nums = numberIds(tok, THOUSANDTHS, 1000)
            ?: return groupProbs(logits, gridReadoutGroups(tok, gridQuestion(EDGE_CODE_GRID), EDGE_CODE_GRID.size), null, null, false)
        return groupProbs(logits, nums, null, null, false)
    }

    private fun presenceProbs(logits: FloatArray, tok: LfmTokenizer, calibration: Calibration?): DoubleArray =
        groupProbs(logits, readoutGroups(tok, Noul("", null, null)), calibration, "noul", true)

    // ------------------------------------------------------------ Antworten

    /**
     * Antworten zu den Frage-Namen; verbraucht slotLogits[0..] in Frage-Reihenfolge
     * (wie api.answer).
     */
    fun answersNamed(tok: LfmTokenizer, questions: List<NamedQ>, slotLogits: Array<FloatArray>,
                     calibration: Calibration?, scoreReadout: String): JSONObject {
        val out = JSONObject()
        var s = 0
        for ((name, q) in questions) {
            val a = JSONObject()
            when (q) {
                is Noul -> {
                    val p = presenceProbs(slotLogits[s++], tok, calibration)
                    a.put("type", "noul").put("noul", p[0])
                }
                is Choice -> {
                    if (q.multiple) {
                        val probs = ArrayList<Double>(q.criteria.size)
                        for (opt in q.criteria.indices) {
                            val p = presenceProbs(slotLogits[s++], tok, calibration)
                            probs.add(calibration?.calibrate("multiple", p[0]) ?: p[0])
                        }
                        val choices = JSONArray()
                        val probMap = JSONObject()
                        for ((i, opt) in q.criteria.withIndex()) {
                            probMap.put(opt.first, probs[i])
                            if (probs[i] >= 0.5) choices.put(opt.first)
                        }
                        a.put("type", "choice").put("choices", choices).put("probabilities", probMap)
                    } else {
                        val p = groupProbs(slotLogits[s++], readoutGroups(tok, q), calibration, choiceBucket(q.criteria.size), true)
                        var best = 0
                        for (i in p.indices) if (p[i] > p[best]) best = i
                        val probMap = JSONObject()
                        for ((i, opt) in q.criteria.withIndex()) probMap.put(opt.first, p[i])
                        a.put("type", "choice").put("choice", q.criteria[best].first)
                            .put("confidence", p[best]).put("probabilities", probMap)
                    }
                }
                is Score -> {
                    val p = groupProbs(slotLogits[s++], readoutGroups(tok, q), calibration, "score:${q.criteria.size}", true)
                    val level: Int = when (scoreReadout) {
                        "mean" -> {
                            var acc = 0.0
                            for ((i, v) in p.withIndex()) acc += i * v
                            maxOf(0, min(q.criteria.size - 1, Math.round(acc).toInt()))
                        }
                        "argmax" -> {
                            var best = 0
                            for (i in p.indices) if (p[i] > p[best]) best = i
                            best
                        }
                        else -> { // median (Standard)
                            var mass = 0.0
                            var lvl = q.criteria.size - 1
                            for ((i, v) in p.withIndex()) {
                                mass += v
                                if (mass >= 0.5) { lvl = i; break }
                            }
                            lvl
                        }
                    }
                    var best = 0
                    for (i in p.indices) if (p[i] > p[best]) best = i
                    val probMap = JSONObject()
                    val legend = JSONObject()
                    for ((i, c) in q.criteria.withIndex()) { probMap.put(i.toString(), p[i]); legend.put(i.toString(), c) }
                    a.put("type", "score").put("score", level).put("confidence", p[best])
                        .put("probabilities", probMap).put("legend", legend)
                }
                is Estimate -> {
                    val dist = estimateDistribution(tok, slotLogits[s++], q)
                    val read = gridRead(q.grid, dist) { share -> calibration?.level(share) ?: share }
                    a.put("type", "number").put("number", read["median"])
                        .put("interval", JSONArray().put(read["p10"]).put(read["p90"]))
                    if (q.unit != null) a.put("unit", q.unit)
                }
                is Box -> {
                    val foundRaw = presenceProbs(slotLogits[s++], tok, calibration)[0]
                    val edges = (0 until 4).map { edgeDistribution(slotLogits[s++], tok) }
                    a.put("type", "box")
                    val f = calibration?.calibrate("found:box", foundRaw) ?: foundRaw
                    if (f < 0.5) {
                        a.put("box", JSONObject.NULL).put("found", f).put("confidence", 0.0)
                    } else {
                        val box = medians(edges)
                        val conf = boxConfidence(edges, box)
                        a.put("box", box).put("found", f)
                            .put("confidence", calibration?.calibrate("box", conf) ?: conf)
                    }
                }
                is Boxes -> {
                    val founds = ArrayList<Double>(q.blocks)
                    val blockEdges = ArrayList<List<DoubleArray>>(q.blocks)
                    for (b in 0 until q.blocks) {
                        founds.add(presenceProbs(slotLogits[s++], tok, calibration)[0])
                        blockEdges.add((0 until 4).map { edgeDistribution(slotLogits[s++], tok) })
                    }
                    val presence = ArrayList<Double>(q.blocks)
                    var below = 1.0
                    for (pYes in founds) {
                        below = min(below, calibration?.calibrate("found:boxes", pYes) ?: pYes)
                        presence.add(below)
                    }
                    val list = ArrayList<Pair<Int, JSONObject>>()
                    for ((b, f) in presence.withIndex()) {
                        if (f < 0.5) break
                        val box = medians(blockEdges[b])
                        val conf = boxConfidence(blockEdges[b], box)
                        list.add(b to JSONObject().put("box", box).put("found", f)
                            .put("confidence", calibration?.calibrate("boxes", conf) ?: conf))
                    }
                    list.sortWith(compareBy(
                        { it.second.getJSONObject("box").getDouble("left") },
                        { it.second.getJSONObject("box").getDouble("top") }))
                    val boxes = JSONArray()
                    for ((_, o) in list) boxes.put(o)
                    a.put("type", "boxes").put("boxes", boxes).put("count", list.size)
                        .put("at_least", list.isNotEmpty() && list.size == presence.size)
                        .put("presence", JSONArray(presence))
                }
            }
            out.put(name, a)
        }
        return out
    }

    private fun choiceBucket(k: Int): String = when {
        k <= 3 -> "choice:small"
        k <= 8 -> "choice:mid"
        k <= 26 -> "choice:large"
        else -> "choice:inventory"
    }

    /** Estimate-Distribution über die Grid-Werte (Zahlentokens oder Codes). */
    private fun estimateDistribution(tok: LfmTokenizer, logits: FloatArray, q: Estimate): DoubleArray {
        val nums = numberIds(tok, q.grid, 1)
        return if (nums != null) {
            groupProbs(logits, nums, null, null, false)
        } else {
            val slotQ = gridQuestion(q.grid)
            groupProbs(logits, gridReadoutGroups(tok, slotQ, q.grid.size), null, null, false)
        }
    }

    /** grid.read: mean/median/p10/p90 auf den Grid-Werten, Quantile über `level`. */
    private fun gridRead(grid: Grid, probs: DoubleArray, level: (Double) -> Double): JSONObject {
        val total = probs.sum().coerceAtLeast(1e-12)
        var mean = 0.0
        for ((i, p) in probs.withIndex()) mean += grid.values[i] * (p / total)
        val out = JSONObject().put("mean", mean)
        val at = listOf("p10" to 0.1, "median" to 0.5, "p90" to 0.9)
        var mass = 0.0
        for ((i, p) in probs.withIndex()) {
            mass += p / total
            for ((key, quant) in at) {
                if (!out.has(key) && mass >= level(quant)) out.put(key, grid.values[i])
            }
        }
        for ((key, _) in at) if (!out.has(key)) out.put(key, grid.values[grid.size - 1])
        return out
    }
}
