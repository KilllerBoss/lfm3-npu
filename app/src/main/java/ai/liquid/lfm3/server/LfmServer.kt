package ai.liquid.lfm3.server

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import ai.liquid.lfm3.BuildConfig
import ai.liquid.lfm3.engine.EngineHost
import ai.liquid.lfm3.engine.ModelCatalog
import ai.liquid.lfm3.engine.ModelKind
import ai.liquid.lfm3.engine.ModelManager
import ai.liquid.lfm3.engine.ModelTasks
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import java.io.PipedInputStream
import java.io.PipedOutputStream
import java.util.concurrent.CountDownLatch

/**
 * Lokaler HTTP-Server, über den die im Vollbild laufende HTML-App das
 * on-device LFM3-Modell nutzt. Bindet nur auf 127.0.0.1 (kein Externe-Zugriff).
 *
 * Endpunkte:
 *   GET  /status                  – Server-/Modell-Status
 *   GET  /models                  – Modellkatalog mit Installations-/Ladezustand
 *   POST /load  {id}              – installiertes Modell laden
 *   POST /v1/chat/completions     – OpenAI-kompatible Chat-API (Stream via SSE)
 *   POST /stop                    – laufende Generierung abbrechen
 *   GET  /app.html                – die zuletzt gestartete HTML-App
 */
class LfmServer(private val appCtx: Context, port: Int) : NanoHTTPD("127.0.0.1", port) {

    @Volatile var currentHtml: String? = null

    private val startTime = System.currentTimeMillis()

    // ================================================================ Routing

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri ?: "/"
        return try {
            if (session.method == Method.OPTIONS) {
                newFixedLengthResponse(Response.Status.NO_CONTENT, "text/plain", "").withCors()
            } else when (uri) {
                "/", "/status" -> statusJson()
                "/models" -> modelsJson()
                "/load" -> if (session.method == Method.POST) loadJson(session)
                           else jsonError(405, "Nur POST")
                "/stop" -> { EngineHost.stopGeneration(); okJson() }
                "/v1/chat/completions" ->
                    if (session.method == Method.POST) chatCompletions(session)
                    else jsonError(405, "Nur POST")
                "/v1/decide" ->
                    if (session.method == Method.POST) decideCompletions(session)
                    else jsonError(405, "Nur POST")
                "/app.html" -> serveApp()
                else -> jsonError(404, "Unbekannter Endpunkt: $uri")
            }
        } catch (t: Throwable) {
            jsonError(500, "Server-Fehler: ${t.message ?: t.javaClass.simpleName}")
        }
    }

    // ================================================================ Status

    private fun statusJson(): Response {
        val model = EngineHost.currentModel
        val modelJson = if (model != null) JSONObject()
            .put("id", model.id)
            .put("name", model.displayName)
            .put("kind", model.kind.name)
            .put("precision", EngineHost.currentPrecision)
            .put("image_input", model.needsVision)
            .put("audio_input", false)
            .put("single_pass", EngineHost.engine?.isSinglePass == true)
        else JSONObject.NULL
        val o = JSONObject()
            .put("ok", true)
            .put("app", "LFM3 NPU")
            .put("version", BuildConfig.VERSION_NAME)
            .put("backend", if (EngineHost.isModelLoaded) EngineHost.backendLabel else JSONObject.NULL)
            .put("uptime_ms", System.currentTimeMillis() - startTime)
            .put("generating", EngineHost.isGenerating)
            .put("model", modelJson)
            .put("endpoints", JSONArray()
                .put("GET /status").put("GET /models").put("POST /load")
                .put("POST /v1/chat/completions").put("POST /v1/decide").put("POST /stop"))
        return newFixedLengthResponse(Response.Status.OK, "application/json", o.toString()).withCors()
    }

    private fun modelsJson(): Response {
        val arr = JSONArray()
        for (m in ModelCatalog.MODELS) {
            val prec = ModelTasks.precisionInfo(appCtx, m)
            val ds = ModelTasks.downloadState(m, appCtx)
            val o = JSONObject()
                .put("id", m.id)
                .put("name", m.displayName)
                .put("kind", m.kind.name)
                .put("desc", m.desc)
                .put("precision", prec.id)
                .put("size_bytes", m.totalBytes(prec.id))
                .put("installed", ModelManager.isInstalled(appCtx, m, prec))
                .put("loaded", EngineHost.currentModel?.id == m.id)
                .put("image_input", m.needsVision)
                .put("audio_input", false)
                .put("single_pass", m.kind == ModelKind.DECISION)
            if (ds != null) {
                o.put("downloading", true).put("download_pct", ds.pct)
            }
            arr.put(o)
        }
        val all = JSONObject()
            .put("models", arr)
            .put("loaded", EngineHost.currentModel?.id ?: JSONObject.NULL)
        return newFixedLengthResponse(Response.Status.OK, "application/json", all.toString()).withCors()
    }

    private fun loadJson(session: IHTTPSession): Response {
        if (EngineHost.isGenerating) EngineHost.stopGeneration()
        val req = try { JSONObject(readBody(session)) } catch (t: Throwable) {
            return jsonError(400, "Ungültiges JSON: ${t.message}")
        }
        val id = req.optString("id", "")
        val m = ModelCatalog.byId(id) ?: return jsonError(404, "Unbekanntes Modell: $id")
        val precId = req.optString("precision", ModelTasks.selectedPrecision(appCtx, m))
        val prec = m.precision(precId) ?: return jsonError(400, "Unbekannte Präzision: $precId")
        if (!ModelManager.isInstalled(appCtx, m, prec)) {
            return jsonError(400, "Modell '${m.id}' [${prec.id}] ist nicht installiert – bitte im Launcher herunterladen")
        }
        val latch = CountDownLatch(1)
        var err: String? = null
        ModelTasks.load(appCtx, m, prec) { e -> err = e; latch.countDown() }
        latch.await()
        return if (err == null) okJson() else jsonError(500, "Ladefehler: $err")
    }

    // ================================================================ Chat-API

    private class ParsedMsgs(val system: String?, val history: List<Pair<String, String>>,
                             val image: Bitmap?)

    private fun chatCompletions(session: IHTTPSession): Response {
        if (EngineHost.engine == null) return jsonError(503,
            "Kein Modell geladen. Öffne die LFM3-App und lade im Launcher ein Modell.")
        if (EngineHost.engine?.isSinglePass == true) return jsonError(400,
            "Das geladene Modell (d1-2B) ist ein Ein-Pass-Entscheidungsmodell ohne Textgenerierung " +
            "(kein KV-Cache). Für Boxen/Zählen/Auswahl aus Bildern nutze POST /v1/decide (System-One); " +
            "für die Chat-API bitte LFM3-VL-2B (Vision+JSON) oder LFM3-1.7B (Text) laden.")

        val req = try { JSONObject(readBody(session)) } catch (t: Throwable) {
            return jsonError(400, "Ungültiges JSON: ${t.message}")
        }
        val msgs = req.optJSONArray("messages")
            ?: return jsonError(400, "'messages' (Array) fehlt")
        val parsed = parseMessages(msgs)
        if (parsed.history.isEmpty()) return jsonError(400, "'messages' enthält keine user/assistant-Nachricht")
        if (parsed.history.last().first != "user") return jsonError(400, "Die letzte Nachricht muss 'user' sein")
        if (parsed.image != null && EngineHost.currentModel?.needsVision != true) {
            return jsonError(400, "Das geladene Modell kann keine Bilder – lade ein Vision-Modell (z.B. LFM3-VL-2B)")
        }

        val stream = req.optBoolean("stream", false)
        val maxTokens = req.optInt("max_tokens", req.optInt("max_new_tokens", 512))
        val temperature = req.optDouble("temperature", 0.7).toFloat()
        val reasoning = req.optBoolean("reasoning", req.optBoolean("thinking", false))
        val model = EngineHost.currentModel?.id ?: "unknown"

        // Auto-Cancel: laufende Generierung wird durch neue Anfrage ersetzt
        if (EngineHost.isGenerating) EngineHost.stopGeneration()

        return if (stream) streamChat(parsed, maxTokens, temperature, reasoning, model)
        else blockingChat(parsed, maxTokens, temperature, reasoning, model)
    }

    /** Nicht-streamende Variante: wartet auf das vollständige Ergebnis. */
    private fun blockingChat(
        parsed: ParsedMsgs, maxTokens: Int, temperature: Float,
        reasoning: Boolean, model: String
    ): Response {
        val latch = CountDownLatch(1)
        val answer = StringBuilder()
        val think = StringBuilder()
        var stats: ai.liquid.lfm3.engine.GenStats? = null
        var promptTokens = 0
        var error: String? = null

        EngineHost.chatApi(
            system = parsed.system, history = parsed.history, reasoning = reasoning,
            maxTokens = maxTokens, temperature = temperature, image = parsed.image,
            onDelta = { _, _ -> },
            onDone = { a, r, s, pt ->
                answer.append(a); think.append(r); stats = s; promptTokens = pt
                latch.countDown()
            },
            onError = { e2 -> error = e2; latch.countDown() }
        )
        latch.await()

        if (error != null) return jsonError(500, "Generierung fehlgeschlagen: $error")
        val st = stats
        val msg = JSONObject()
            .put("role", "assistant")
            .put("content", answer.toString())
        if (think.isNotBlank()) msg.put("reasoning_content", think.toString())
        val o = JSONObject()
            .put("id", "chatcmpl-${System.currentTimeMillis()}")
            .put("object", "chat.completion")
            .put("created", System.currentTimeMillis() / 1000)
            .put("model", model)
            .put("choices", JSONArray().put(JSONObject()
                .put("index", 0)
                .put("message", msg)
                .put("finish_reason", if (st?.truncated == true) "length" else "stop")))
        if (st != null) {
            o.put("usage", JSONObject()
                .put("prompt_tokens", promptTokens)
                .put("completion_tokens", st.tokens)
                .put("total_tokens", promptTokens + st.tokens))
            o.put("stats", JSONObject()
                .put("backend", EngineHost.backendLabel)
                .put("prefill_ms", st.prefillMs)
                .put("decode_ms", st.decodeMs)
                .put("tokens_per_second", st.tokensPerSec.toDouble())
                .put("context_used", st.contextUsed))
        }
        return newFixedLengthResponse(Response.Status.OK, "application/json", o.toString()).withCors()
    }

    /** Streamende Variante (SSE): Token für Token als "data: {...}"-Zeilen. */
    private fun streamChat(
        parsed: ParsedMsgs, maxTokens: Int, temperature: Float,
        reasoning: Boolean, model: String
    ): Response {
        val pin = PipedInputStream(128 * 1024)
        val pout = PipedOutputStream()
        try { pout.connect(pin) } catch (t: Throwable) { return jsonError(500, "${t.message}") }

        val id = "chatcmpl-${System.currentTimeMillis()}"
        val created = System.currentTimeMillis() / 1000

        fun chunk(delta: JSONObject, finish: String? = null): String {
            val c = JSONObject()
                .put("id", id).put("object", "chat.completion.chunk")
                .put("created", created).put("model", model)
                .put("choices", JSONArray().put(JSONObject()
                    .put("index", 0).put("delta", delta)
                    .put("finish_reason", finish ?: JSONObject.NULL)))
            return "data: ${c}\n\n"
        }

        EngineHost.chatApi(
            system = parsed.system, history = parsed.history, reasoning = reasoning,
            maxTokens = maxTokens, temperature = temperature, image = parsed.image,
            onDelta = { text, inThink ->
                val d = JSONObject()
                if (inThink) d.put("reasoning_content", text) else d.put("content", text)
                try { pout.write(chunk(d).toByteArray(Charsets.UTF_8)); pout.flush() } catch (_: Throwable) {}
            },
            onDone = { _, _, st, _ ->
                try {
                    if (st != null && st.tokens > 0) {
                        val u = JSONObject().put("prompt_tokens", 0)
                            .put("completion_tokens", st.tokens)
                            .put("total_tokens", st.tokens)
                        pout.write("data: ${JSONObject().put("id", id).put("usage", u)}\n\n".toByteArray(Charsets.UTF_8))
                    }
                    pout.write(chunk(JSONObject(), "stop").toByteArray(Charsets.UTF_8))
                    pout.write("data: [DONE]\n\n".toByteArray(Charsets.UTF_8))
                } catch (_: Throwable) {}
                try { pout.close() } catch (_: Throwable) {}
            },
            onError = { e2 ->
                try {
                    val err = JSONObject().put("error", JSONObject()
                        .put("message", "Generierung fehlgeschlagen: $e2").put("type", "server_error"))
                    pout.write("data: ${err}\n\n".toByteArray(Charsets.UTF_8))
                    pout.write("data: [DONE]\n\n".toByteArray(Charsets.UTF_8))
                } catch (_: Throwable) {}
                try { pout.close() } catch (_: Throwable) {}
            }
        )

        return newChunkedResponse(Response.Status.OK, "text/event-stream", pin).withCors()
    }

    // ================================================================ Decide-API (d1)

    /** Max. State-Länge (Zeichen) – schützt Speicher & Kontext (d1: 128k-Tokens-Limit). */
    private val MAX_STATE_CHARS = 24_000
    /** Max. Bilder pro Decision (d1 concatVision). */
    private val MAX_DECIDE_IMAGES = 6

    /**
     * POST /v1/decide – System-One-Fragen über ein d1-Modell (ein Forward-Pass):
     * {"state": "…" | null, "image": "data:image/…;base64,…", "images": [...],
     *  "score_readout": "median",
     *  "questions": {"name": {"type": "box|boxes|number|choice|score|noul",
     *                          "instructions": "…", "criteria": …, "range": …, "max": …}}}
     * -> {"ok":true, "answers": {"name": {typisierte Antwort}}, "usage": {…}}
     */
    private fun decideCompletions(session: IHTTPSession): Response {
        if (EngineHost.engine == null) return jsonError(503,
            "Kein Modell geladen. Öffne die LFM3-App und lade im Launcher ein Modell.")
        if (EngineHost.engine?.isSinglePass != true) return jsonError(400,
            "/v1/decide braucht ein d1-Entscheidungsmodell (Ein-Pass). " +
            "Bitte im Launcher d1-2B laden – für Chat/Vision über die Chat-API " +
            "bitte LFM3-VL-2B oder LFM3-1.7B laden.")
        val req = try { JSONObject(readBody(session)) } catch (t: Throwable) {
            return jsonError(400, "Ungültiges JSON: ${t.message}")
        }
        val questions = req.optJSONObject("questions")
            ?: return jsonError(400, "'questions' (Objekt) fehlt: " +
                "{\"name\": {\"type\": \"box\", \"instructions\": \"Wo ist …?\"}}")
        val state = if (req.has("state") && !req.isNull("state")) req.optString("state") else null
        if (state != null && state.length > MAX_STATE_CHARS) {
            return jsonError(400, "State ist zu lang: ${state.length} Zeichen (Maximum $MAX_STATE_CHARS). " +
                "Bitte kürzen – d1 packt State + Fragen in einen Forward-Pass.")
        }
        val images = ArrayList<Bitmap>()
        val urls = ArrayList<String>()
        if (req.has("image") && !req.isNull("image")) urls.add(req.optString("image"))
        req.optJSONArray("images")?.let { arr ->
            for (i in 0 until arr.length()) urls.add(arr.optString(i))
        }
        if (urls.size > MAX_DECIDE_IMAGES) {
            return jsonError(400, "Zu viele Bilder: ${urls.size} (Maximum $MAX_DECIDE_IMAGES pro Decision)")
        }
        for (u in urls) {
            val bmp = decodeDataImage(u)
                ?: return jsonError(400, "Bild konnte nicht dekodiert werden (data-URL mit base64 erwartet)")
            images.add(bmp)
        }
        val scoreReadout = req.optString("score_readout", "median")
        if (scoreReadout !in setOf("mean", "median", "argmax")) {
            return jsonError(400, "score_readout ist mean, median oder argmax – nicht '$scoreReadout'")
        }

        val latch = CountDownLatch(1)
        var result: JSONObject? = null
        var error: String? = null
        EngineHost.decideApi(
            ctx = appCtx, state = state, images = images, questions = questions,
            scoreReadout = scoreReadout,
            onDone = { r, _ -> result = r; latch.countDown() },
            onError = { e2 -> error = e2; latch.countDown() }
        )
        latch.await()
        if (error != null) return jsonError(400, "Decision fehlgeschlagen: $error")
        return newFixedLengthResponse(Response.Status.OK, "application/json",
            (result ?: JSONObject()).put("ok", true).toString()).withCors()
    }

    // ================================================================ App-HTML

    private fun serveApp(): Response {
        val html = currentHtml
        return if (html != null) {
            newFixedLengthResponse(Response.Status.OK, "text/html; charset=utf-8", html).withCors()
        } else {
            val status = if (EngineHost.isModelLoaded)
                "Modell geladen: <b>${EngineHost.currentModel?.displayName}</b> (${EngineHost.backendLabel})"
            else "<b style=\"color:#FF6B6B\">Kein Modell geladen</b> – im Launcher über „Modelle“ laden."
            newFixedLengthResponse(Response.Status.OK, "text/html; charset=utf-8", """
                <!DOCTYPE html><html lang="de"><head><meta charset="utf-8">
                <meta name="viewport" content="width=device-width,initial-scale=1">
                <title>LFM3 NPU</title></head>
                <body style="background:#0B0F19;color:#F2F4FA;font-family:sans-serif;padding:24px">
                <h2>LFM3 NPU – Launcher</h2>
                <p>Hier läuft deine HTML-App, sobald du im Launcher HTML-Code einfügst und auf
                <b>„HTML im Vollbild starten“</b> tippst.</p>
                <p>$status</p>
                <p style="color:#9AA3B8">API: <code>GET /status</code>, <code>GET /models</code>,
                <code>POST /v1/chat/completions</code> (OpenAI-Format)</p>
                </body></html>
            """.trimIndent()).withCors()
        }
    }

    // ================================================================ Helpers

    private fun readBody(session: IHTTPSession): String {
        val files = HashMap<String, String>()
        session.parseBody(files)
        return files["postData"] ?: ""
    }

    private fun Response.withCors(): Response {
        addHeader("Access-Control-Allow-Origin", "*")
        addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization")
        return this
    }

    private fun okJson(): Response = newFixedLengthResponse(Response.Status.OK,
        "application/json", "{\"ok\":true}").withCors()

    private fun jsonError(code: Int, message: String): Response =
        newFixedLengthResponse(Response.Status.lookup(code) ?: Response.Status.INTERNAL_ERROR,
            "application/json",
            JSONObject().put("error", JSONObject().put("message", message)
                .put("type", "lfm3_error").put("code", code)).toString()).withCors()

    /** messages-Array (String- oder Parts-Content) -> System/History/Bitmap. */
    private fun parseMessages(arr: JSONArray): ParsedMsgs {
        val sys = StringBuilder()
        val hist = ArrayList<Pair<String, String>>()
        var img: Bitmap? = null
        for (i in 0 until arr.length()) {
            val m = arr.optJSONObject(i) ?: continue
            val role = m.optString("role")
            var text = ""
            when (val content = m.opt("content")) {
                is String -> text = content
                is JSONArray -> {
                    val sb = StringBuilder()
                    for (j in 0 until content.length()) {
                        val part = content.optJSONObject(j) ?: continue
                        when (part.optString("type")) {
                            "text" -> sb.append(part.optString("text"))
                            "image_url" -> {
                                if (img == null) {
                                    val url = part.optJSONObject("image_url")?.optString("url")
                                        ?: part.optString("image_url")
                                    img = decodeDataImage(url)
                                }
                            }
                            // "input_audio" & Co. werden bewusst ignoriert (nicht unterstützt)
                        }
                    }
                    text = sb.toString()
                }
            }
            when (role) {
                "system" -> { if (sys.isNotEmpty()) sys.append("\n"); sys.append(text) }
                "user" -> hist.add("user" to text)
                "assistant" -> hist.add("assistant" to text)
            }
        }
        val sysStr = sys.toString().trim()
        return ParsedMsgs(sysStr.ifEmpty { null }, hist, img)
    }

    private fun decodeDataImage(url: String?): Bitmap? {
        if (url.isNullOrBlank()) return null
        return try {
            val b64 = if (url.startsWith("data:")) url.substringAfter("base64,", "") else url
            if (b64.isBlank()) return null
            val bytes = Base64.decode(b64, Base64.DEFAULT)
            // Grenzen lesen, dann mit inSampleSize dekodieren – verhindert riesige
            // Bitmaps (Speicher-Spitzen/OOM bei 50-MP-Kamerafotos).
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 2048) sample *= 2
            val opts = BitmapFactory.Options().apply { inSampleSize = sample }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
        } catch (_: Throwable) { null }
    }
}

/**
 * Startet den Server mit Port-Fallback (8787 → 8790 → zufällig) und hält
 * die Instanz für die gesamte App-Laufzeit.
 */
object ServerHost {
    @Volatile var server: LfmServer? = null
        private set

    val port: Int get() = server?.listeningPort?.takeIf { it > 0 } ?: 8787
    val baseUrl: String get() = "http://127.0.0.1:$port"
    val isRunning: Boolean get() = server != null

    fun start(appCtx: Context) {
        if (server != null) return
        synchronized(this) {
            if (server != null) return
            for (p in intArrayOf(8787, 8788, 8789, 8790, 0)) {
                try {
                    val s = LfmServer(appCtx.applicationContext, p)
                    s.start(15_000, true)
                    server = s
                    android.util.Log.i("LFM3-Server", "HTTP-Server auf Port ${s.listeningPort} gestartet")
                    return
                } catch (t: Throwable) {
                    android.util.Log.w("LFM3-Server", "Port $p nicht möglich: ${t.message}")
                }
            }
        }
    }
}
