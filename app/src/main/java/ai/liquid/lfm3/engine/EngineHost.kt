package ai.liquid.lfm3.engine

import android.content.Context
import android.content.SharedPreferences
import android.graphics.Bitmap
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Zentrale Instanz: hält geladenes Modell + Tokenizer, Einstellungen und
 * den Worker-Thread für alle Engine-Operationen.
 */
object EngineHost {

    class Msg(
        @JvmField val role: String,           // "user" | "assistant"
        @JvmField var text: String,
        @JvmField var think: String = "",
        @JvmField var stats: String = "",
        @JvmField val bitmap: Bitmap? = null
    )

    private val executor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread(r, "lfm3-engine").apply { priority = Thread.NORM_PRIORITY + 1 }
    }

    @Volatile var engine: OrtEngine? = null
        private set
    @Volatile var tokenizer: LfmTokenizer? = null
        private set
    @Volatile var currentModel: ModelInfo? = null
        private set
    @Volatile var currentPrecision: String? = null
        private set
    @Volatile var isGenerating: Boolean = false
        private set
    @Volatile var loadError: String? = null

    private var cancelFlag = AtomicBoolean(false)
    private var genFuture: Future<*>? = null

    // ------------------------------------------------------------- Einstellungen
    private lateinit var prefs: SharedPreferences

    fun init(ctx: Context) {
        if (this::prefs.isInitialized) return
        prefs = ctx.applicationContext.getSharedPreferences("lfm3", Context.MODE_PRIVATE)
    }

    var systemPrompt: String
        get() = if (::prefs.isInitialized) prefs.getString("system",
            "Du bist ein hilfreicher KI-Assistent. Antworte kurz und präzise auf Deutsch.")!!
        else ""
        set(v) = prefs.edit().putString("system", v).apply()

    var thinkingMode: Boolean
        get() = if (::prefs.isInitialized) prefs.getBoolean("think", true) else true
        set(v) = prefs.edit().putBoolean("think", v).apply()

    var npuFallback: Boolean
        get() = if (::prefs.isInitialized) prefs.getBoolean("fallback", true) else true
        set(v) = prefs.edit().putBoolean("fallback", v).apply()

    var maxContext: Int
        get() = if (::prefs.isInitialized) prefs.getInt("ctx", 4096) else 4096
        set(v) = prefs.edit().putInt("ctx", v).apply()

    var temperature: Float
        get() = if (::prefs.isInitialized) prefs.getFloat("temp", 0.1f) else 0.1f
        set(v) = prefs.edit().putFloat("temp", v).apply()

    val backendLabel: String
        get() = engine?.backendName ?: "—"

    val isModelLoaded: Boolean
        get() = engine?.isLoaded == true && tokenizer != null

    val activeModelLabel: String
        get() {
            val m = currentModel ?: return "LFM3 NPU"
            val p = currentPrecision ?: return m.displayName
            return "${m.displayName} · $p"
        }

    // ---------------------------------------------------------------- Laden

    /**
     * Lädt ein Modell (Tokenizer + Decoder + optional Vision-Encoder).
     * Läuft im Engine-Thread; Callbacks auf dem aufrufenden Thread.
     */
    fun loadModel(
        ctx: Context,
        model: ModelInfo,
        prec: PrecisionInfo,
        log: (String) -> Unit,
        onDone: (error: String?) -> Unit
    ): Future<*> = executor.submit {
        isGenerating = false
        cancelFlag.set(true)
        try {
            engine?.close()
        } catch (_: Throwable) {}
        engine = null
        tokenizer = null
        currentModel = null
        loadError = null

        try {
            val dir = ModelManager.modelDir(ctx, model.id)
            val tokFile = File(dir, "tokenizer.json")
            if (!tokFile.isFile) throw IllegalStateException("tokenizer.json fehlt – bitte neu herunterladen")

            log("Tokenizer wird gelesen (${ModelCatalog.humanBytes(tokFile.length())})…")
            val tokJson = tokFile.readText()
            val tok = LfmTokenizer(tokJson)
            log("$tok")

            val decoderPath = File(dir, prec.decoderPath)
            if (!decoderPath.isFile) throw IllegalStateException("Modell-Datei fehlt: ${prec.decoderPath}")

            val visionPath = prec.visionPath?.let { vp ->
                val vf = File(dir, vp)
                if (vf.isFile) vf.absolutePath else null
            }

            val e = OrtEngine()
            log("Session startet (QNN/HTP-Anfrage)…")
            val backend = e.load(
                decoderPath.absolutePath,
                useNpu = true,
                allowFallback = npuFallback,
                visionModelPath = visionPath,
                log = log
            )
            engine = e
            tokenizer = tok
            currentModel = model
            currentPrecision = prec.id
            loadError = null
            log("Geladen: ${model.displayName} [${prec.id}] via $backend")
            onDone(null)
        } catch (t: Throwable) {
            loadError = t.message ?: t.toString()
            log("Fehler beim Laden: $loadError")
            try { engine?.close() } catch (_: Throwable) {}
            engine = null
            onDone(loadError)
        }
    }

    // ---------------------------------------------------------------- Chat

    /**
     * Vollständige Generierung für eine Chat-Anfrage (Reset + vollständiger Prefill).
     * onDelta wird im Engine-Thread aufgerufen (UI muss selbst auf Main posten).
     */
    fun chat(
        history: List<Msg>,
        image: Bitmap?,
        onDelta: (text: String, inThink: Boolean) -> Unit,
        onStats: (GenStats) -> Unit,
        onError: (String) -> Unit
    ): Future<*> {
        val e = engine ?: run { onError("Kein Modell geladen"); return executor.submit {} }
        val tok = tokenizer ?: run { onError("Kein Tokenizer"); return executor.submit {} }
        val model = currentModel

        if (isGenerating) {
            onError("Es läuft bereits eine Generierung")
            return executor.submit {}
        }

        val histPairs = history.map { it.role to it.text }
        val prompt = ChatTemplate.render(systemPrompt, histPairs, thinkingMode)
        val ids = tok.encode(prompt)
        if (ids.isEmpty()) { onError("Leerer Prompt"); return executor.submit {} }

        val cfg = GenConfig(
            maxTokens = minOf(2048, maxOf(256, maxContext / 2)),
            temperature = temperature,
            topK = 50,
            topP = 0.95f,
            repPenalty = 1.05f,
            repWindow = 128,
            maxContext = maxContext
        )

        cancelFlag = AtomicBoolean(false)
        val flag = cancelFlag
        isGenerating = true

        genFuture = executor.submit {
            try {
                e.reset()
                var inThink = thinkingMode
                val thinkBuf = StringBuilder()
                val msgBuf = StringBuilder()
                var lastSpecial = 0

                val stats = e.generate(
                    ids, cfg, tok,
                    if (model?.needsVision == true) image else null,
                    onDelta = { text ->
                        if (inThink) {
                            thinkBuf.append(text)
                            onDelta(text, true)
                        } else {
                            msgBuf.append(text)
                            onDelta(text, false)
                        }
                    },
                    onSpecial = { id ->
                        lastSpecial = id
                        if (id == ChatTemplate.TOKEN_THINK_END && inThink) {
                            inThink = false
                        }
                    },
                    cancel = flag,
                    log = { /* Engine-Log */ }
                )

                // Sicherheitsnetz: falls </think> nie kam, aber Text seit <think> leer blieb
                if (msgBuf.isEmpty() && thinkBuf.isNotEmpty() && inThink) {
                    onDelta("", false)
                }
                isGenerating = false
                onStats(stats)
            } catch (t: Throwable) {
                isGenerating = false
                onError(t.message ?: t.toString())
            }
        }
        return genFuture!!
    }

    fun stopGeneration() {
        cancelFlag.set(true)
    }

    // ---------------------------------------------------------------- Einzelausführung (Agent)

    /** Ein einzelner Prompt -> Antwort-Text (blockierend, Engine-Thread). */
    fun runOnce(prompt: String, image: Bitmap?, maxTokens: Int = 512): String {
        val e = engine ?: throw IllegalStateException("Kein Modell geladen")
        val tok = tokenizer ?: throw IllegalStateException("Kein Tokenizer")
        val model = currentModel
        val full = ChatTemplate.render(systemPrompt, listOf("user" to prompt), false)
        val ids = tok.encode(full)
        val cfg = GenConfig(
            maxTokens = maxTokens,
            temperature = 0.05f,
            maxContext = maxContext
        )
        e.reset()
        val sb = StringBuilder()
        val flag = AtomicBoolean(false)
        val stats = e.generate(
            ids, cfg, tok,
            if (model?.needsVision == true) image else null,
            onDelta = { sb.append(it) },
            onSpecial = { },
            cancel = flag,
            log = { }
        )
        return sb.toString()
    }

    /** Tool-Unterstützung: Rechner + Uhrzeit (für d1/1.7B Agent). */
    fun tryTool(output: String): String? {
        val regex = Regex("\\{[^{}]*\"tool\"[^{}]*\\}")
        val m = regex.find(output) ?: return null
        return try {
            val json = m.value
            val name = Regex("\"tool\"\\s*:\\s*\"([^\"]+)\"").find(json)?.groupValues?.get(1)
            val arg = Regex("\"(?:expr|arg|input)\"\\s*:\\s*\"([^\"]+)\"").find(json)?.groupValues?.get(1)
            when (name) {
                "calculator" -> {
                    val expr = arg ?: return null
                    val sanitized = expr.replace(Regex("[^0-9+\\-*/().% ]"), "")
                    if (sanitized.isEmpty()) return null
                    val result = evalSimple(sanitized)
                    "Tool-Ergebnis (calculator): $result"
                }
                "time", "clock" -> {
                    val now = java.text.SimpleDateFormat("HH:mm:ss, dd.MM.yyyy",
                        java.util.Locale.GERMANY).format(java.util.Date())
                    "Tool-Ergebnis (time): $now"
                }
                else -> null
            }
        } catch (_: Throwable) { null }
    }

    private fun evalSimple(expr: String): Double = ExprParser(expr).parse()

    /** Minimaler rekursiver Ausdrucks-Parser für + - * / ( ) */
    private class ExprParser(private val expr: String) {
        private var pos = 0

        fun parse(): Double = parseExpr()

        private fun skip() { while (pos < expr.length && expr[pos] == ' ') pos++ }

        private fun parseExpr(): Double {
            var v = parseTerm()
            while (true) {
                skip()
                if (pos < expr.length && (expr[pos] == '+' || expr[pos] == '-')) {
                    val op = expr[pos]; pos++
                    val r = parseTerm()
                    v = if (op == '+') v + r else v - r
                } else return v
            }
        }

        private fun parseTerm(): Double {
            var v = parseAtom()
            while (true) {
                skip()
                if (pos < expr.length && (expr[pos] == '*' || expr[pos] == '/')) {
                    val op = expr[pos]; pos++
                    val r = parseAtom()
                    v = if (op == '*') v * r else v / r
                } else return v
            }
        }

        private fun parseAtom(): Double {
            skip()
            if (pos < expr.length && expr[pos] == '(') {
                pos++
                val v = parseExpr()
                skip()
                if (pos < expr.length && expr[pos] == ')') pos++
                return v
            }
            if (pos < expr.length && expr[pos] == '-') { pos++; return -parseAtom() }
            val start = pos
            while (pos < expr.length && (expr[pos].isDigit() || expr[pos] == '.' || expr[pos] == ',')) pos++
            val s = expr.substring(start, pos).replace(',', '.')
            if (s.isEmpty()) throw IllegalArgumentException("Ungültiger Ausdruck")
            return s.toDouble()
        }
    }
}
