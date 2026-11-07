package ai.liquid.lfm3.engine

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer
import java.util.concurrent.atomic.AtomicBoolean

/** Generierungs-Konfiguration. */
data class GenConfig(
    val maxTokens: Int = 1024,
    val temperature: Float = 0.1f,
    val topK: Int = 50,
    val topP: Float = 0.95f,
    val repPenalty: Float = 1.05f,
    val repWindow: Int = 128,
    val maxContext: Int = 4096
)

/** Ergebnis-Statistiken einer Generierung. */
data class GenStats(
    val tokens: Int,
    val prefillMs: Long,
    val decodeMs: Long,
    val tokensPerSec: Float,
    val truncated: Boolean,
    val contextUsed: Int
)

/**
 * ONNX-Runtime-Engine mit QNN/HTP (Hexagon NPU) und CPU-Fallback.
 *
 * KV-Handling: Nach dem Laden werden die Cache-Inputs (past_*) aus den
 * Session-Metadaten erkannt und als Null-Tensoren initialisiert (statische
 * Dimensionen bleiben, Sequenz-Dimensionen starten bei Länge 0). Die Outputs
 * present.* werden nach jedem Schritt direkt als Inputs des nächsten Schritts
 * wiederverwendet (Zero-Copy): Das OrtSession.Result-Objekt bleibt geöffnet,
 * bis das nächste Ergebnis da ist.
 */
class OrtEngine {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    private var session: OrtSession? = null
    private var visionSession: OrtSession? = null
    private var lastResult: OrtSession.Result? = null
    private var zeroCaches: Map<String, OnnxTensor>? = null
    private var visionFeat: OnnxTensor? = null

    private var cacheInputNames = ArrayList<String>()
    private var visionInputName: String? = null
    private var inputIdsName = "input_ids"
    private var attnName = "attention_mask"
    private var posName = "position_ids"
    private var keepName = "logits_to_keep"

    // Welche Runtime-Inputs deklariert die Session wirklich? (unterschiedlich je Export)
    private var sessionInputs: Set<String> = emptySet()
    private var hasPosInput = true
    private var hasKeepInput = true
    private var hasCacheInputs = true

    var backendName: String = "CPU"
        private set
    var vocabOutSize: Int = 0
        private set

    val isLoaded: Boolean get() = session != null

    /**
     * true = Ein-Pass-Modell ohne KV-Cache (z.B. d1-2B): kann keinen Text
     * Token für Token generieren, sondern antwortet nur über Slot-Readouts.
     */
    val isSinglePass: Boolean get() = session != null && !hasCacheInputs

    // ------------------------------------------------------------------ Laden

    /**
     * Lädt ein ONNX-Modell. Rückgabe: Name des tatsächlich verwendeten Backends.
     * @throws Exception wenn weder QNN (ohne Fallback) noch CPU laden können.
     */
    fun load(
        modelPath: String,
        useNpu: Boolean,
        allowFallback: Boolean,
        visionModelPath: String?,
        log: (String) -> Unit
    ): String {
        close()
        val so = OrtSession.SessionOptions()
        var backend = "CPU"
        if (useNpu) {
            try {
                so.addQnn(
                    mapOf(
                        "backend_path" to "libQnnHtp.so",
                        "htp_performance_mode" to "high_performance"
                    )
                )
                backend = "QNN/HTP (NPU)"
                log("QNN-EP registriert (libQnnHtp.so, high_performance)")
            } catch (t: Throwable) {
                log("QNN nicht verfügbar: ${t.message}")
                if (!allowFallback) throw IllegalStateException("QNN/HTP fehlgeschlagen: ${t.message}", t)
                backend = "CPU (QNN-Fallback)"
            }
        }
        val s = env.createSession(modelPath, so)
        session = s
        backendName = backend

        // Inputs klassifizieren – NUR deklarierte Inputs werden später gefüttert
        // (Exports unterscheiden sich: d1 hat z.B. KEIN logits_to_keep, dafür
        // question_ids/slot_positions/video_features)
        cacheInputNames.clear()
        visionInputName = null
        val inSet = s.inputNames.toSet()
        sessionInputs = inSet
        hasPosInput = inSet.contains(posName)
        hasKeepInput = inSet.contains(keepName)
        var videoName: String? = null
        var featureName: String? = null
        for (name in s.inputNames) {
            val n = name.lowercase()
            when {
                n == inputIdsName || n == attnName || n == posName || n == keepName || n == "cache_position" -> { /* Runtime-Inputs */ }
                n.contains("past") || n.contains("cache") || n.contains("conv") -> cacheInputNames.add(name)
                n.contains("image") || n.contains("pixel") -> if (visionInputName == null) visionInputName = name
                n.contains("video") -> videoName = name
                n.contains("feature") -> featureName = name
                else -> { /* unbekannter Extra-Input (z.B. question_ids) – wird nicht gefüttert */ }
            }
        }
        // Vision-Input: Bild hat Vorrang vor Video/generischen Features
        if (visionInputName == null) visionInputName = featureName ?: videoName
        hasCacheInputs = cacheInputNames.isNotEmpty()
        log("Cache-Inputs: ${cacheInputNames.size}, Vision-Input: $visionInputName, " +
            "position_ids: $hasPosInput, logits_to_keep: $hasKeepInput")
        if (!hasCacheInputs) log("Kein KV-Cache: Ein-Pass-Modell – keine Textgenerierung möglich")

        zeroCaches = buildZeroCaches(s)
        try {
            val outMeta = s.getOutputInfo()
            val lg = outMeta["logits"]
            val shape = (lg?.info as? ai.onnxruntime.TensorInfo)?.shape
            if (shape != null && shape.isNotEmpty()) vocabOutSize = shape.last().toInt()
        } catch (_: Throwable) {}

        if (visionModelPath != null) {
            try {
                val vso = OrtSession.SessionOptions()
                visionSession = env.createSession(visionModelPath, vso)
                log("Vision-Encoder geladen: $visionModelPath")
            } catch (t: Throwable) {
                log("Vision-Encoder fehlgeschlagen: ${t.message}")
                visionSession = null
            }
        }
        return backend
    }

    /** Null-Initialisierung aller Cache-Inputs gemäß deren Metadaten. */
    private fun buildZeroCaches(s: OrtSession): Map<String, OnnxTensor> {
        val map = LinkedHashMap<String, OnnxTensor>()
        val info = s.inputInfo
        for (name in cacheInputNames) {
            val ni = info[name] ?: continue
            val ti = ni.info as? ai.onnxruntime.TensorInfo ?: continue
            val dims = ti.shape ?: continue
            val shape = LongArray(dims.size) { i -> if (dims[i] <= 0) 0L else dims[i] }
            val t = when (ti.type) {
                OnnxJavaType.FLOAT16 -> {
                    val elems = shape.fold(1L) { a, b -> a * b }
                    OnnxTensor.createTensor(env, ShortBuffer.allocate(elems.toInt().coerceAtLeast(1)), shape, OnnxJavaType.FLOAT16)
                }
                else -> {
                    val elems = shape.fold(1L) { a, b -> a * b }
                    OnnxTensor.createTensor(env, FloatBuffer.allocate(elems.toInt().coerceAtLeast(1)), shape)
                }
            }
            map[name] = t
        }
        return map
    }

    // -------------------------------------------------------------- Generierung

    fun generate(
        inputIds: IntArray,
        cfg: GenConfig,
        tokenizer: LfmTokenizer,
        bitmap: android.graphics.Bitmap?,
        onDelta: (String) -> Unit,      // neuer Text (bereits dekodiert)
        onSpecial: (Int) -> Unit,       // Spezial-Token-ID (z.B. <think>-Umschaltung)
        cancel: AtomicBoolean,
        log: (String) -> Unit
    ): GenStats {
        val s = session ?: throw IllegalStateException("Kein Modell geladen")
        if (!hasCacheInputs) throw IllegalStateException(
            "Dieses Modell ist ein Ein-Pass-Entscheidungsmodell (kein KV-Cache, z.B. d1-2B) " +
            "und kann keinen Text Token für Token generieren. " +
            "Bitte ein Chat-Modell laden: LFM3-VL-2B (Vision+JSON) oder LFM3-1.7B (Text).")
        val decoder = LfmTokenizer.StreamDecoder(tokenizer)
        val generated = ArrayList<Int>(cfg.maxTokens)

        var prefillMs = 0L
        var decodeMs = 0L

        // Vision-Features vorbereiten (nur erster Prefill-Chunk)
        if (bitmap != null && visionInputName != null) {
            try {
                val t0 = System.currentTimeMillis()
                val vr = runVision(bitmap)
                visionFeat = adaptVisionToDecoder(vr.feat)
                if (visionFeat !== vr.feat) try { vr.feat.close() } catch (_: Throwable) {}
                log("Vision-Features in ${System.currentTimeMillis() - t0} ms erzeugt " +
                    "(${vr.fw}x${vr.fh} -> ${imageTokenCount(vr.fw, vr.fh)} Tokens)")
            } catch (t: Throwable) {
                log("Vision fehlgeschlagen: ${t.message}")
                visionFeat = null
            }
        }

        // Prefill in 256er-Chunks
        val chunkSize = 256
        var pos = 0
        val prefillStart = System.currentTimeMillis()
        var logits = FloatArray(0)
        while (pos < inputIds.size) {
            if (cancel.get()) throw InterruptedException("Abgebrochen")
            val end = minOf(pos + chunkSize, inputIds.size)
            val chunk = inputIds.copyOfRange(pos, end)
            logits = runStep(chunk, pos, end, feedVision = (pos == 0 && visionFeat != null))
            pos = end
        }
        prefillMs = System.currentTimeMillis() - prefillStart
        // Vision-Features werden nach dem ersten Prefill nicht mehr gebraucht
        closeVisionFeat()

        if (inputIds.size >= cfg.maxContext) {
            return GenStats(0, prefillMs, 0, 0f, true, inputIds.size)
        }

        // Decode-Schleife
        var nextId = sample(logits, cfg, generated)
        val decodeStart = System.currentTimeMillis()
        var truncated = false
        while (true) {
            if (cancel.get()) { decodeMs = System.currentTimeMillis() - decodeStart; break }
            if (nextId in ChatTemplate.STOP_IDS) break
            generated.add(nextId)
            if (nextId == ChatTemplate.TOKEN_THINK_START || nextId == ChatTemplate.TOKEN_THINK_END) {
                // Steuer-Tokens steuern nur den Denk-Modus, erscheinen nicht als Text
                onSpecial(nextId)
            } else {
                val text = decoder.push(nextId)
                if (text.isNotEmpty()) onDelta(text)
            }

            if (generated.size >= cfg.maxTokens) { truncated = true; break }
            if (inputIds.size + generated.size >= cfg.maxContext) { truncated = true; break }

            logits = runStep(intArrayOf(nextId), inputIds.size + generated.size - 1, inputIds.size + generated.size, feedVision = false)
            nextId = sample(logits, cfg, generated)
        }
        decodeMs = System.currentTimeMillis() - decodeStart
        val tail = decoder.flush()
        if (tail.isNotEmpty()) onDelta(tail)

        val toks = generated.size
        val tps = if (decodeMs > 0) toks * 1000f / decodeMs else 0f
        return GenStats(toks, prefillMs, decodeMs, tps, truncated, inputIds.size + toks)
    }

    /** Ein Transformer-Schritt. Rückgabe: Logits der letzten Position [vocab]. */
    private fun runStep(chunk: IntArray, posStart: Int, totalTokens: Int, feedVision: Boolean): FloatArray {
        val s = session ?: throw IllegalStateException("Kein Modell geladen")

        val inputs = HashMap<String, OnnxTensor>()
        val ids = LongArray(chunk.size) { chunk[it].toLong() }
        inputs[inputIdsName] = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, chunk.size.toLong()))
        val mask = LongArray(totalTokens) { 1L }
        inputs[attnName] = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), longArrayOf(1, totalTokens.toLong()))
        if (hasPosInput) {
            val pos = LongArray(chunk.size) { (posStart + it).toLong() }
            inputs[posName] = OnnxTensor.createTensor(env, LongBuffer.wrap(pos), longArrayOf(1, chunk.size.toLong()))
        }
        // logits_to_keep nur füttern, wenn das Modell ihn deklariert (d1 tut das nicht!)
        if (hasKeepInput) {
            inputs[keepName] = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(1L)), longArrayOf(1))
        }

        if (lastResultHasCaches()) {
            for (name in cacheInputNames) {
                val t = lastResult?.get(name)?.orElse(null) as? OnnxTensor
                if (t != null) inputs[name] = t
            }
        } else {
            zeroCaches?.let { zc -> for ((n, t) in zc) inputs[n] = t }
        }
        if (feedVision) {
            visionFeat?.let { vf -> visionInputName?.let { vn -> inputs[vn] = vf } }
        }

        // Sicherheitsnetz: niemals Inputs füttern, die die Session nicht deklariert
        // (sonst: "Unknown input name ..." von ONNX Runtime)
        val iter = inputs.entries.iterator()
        while (iter.hasNext()) {
            val en = iter.next()
            if (en.key !in sessionInputs) {
                if (en.value !== visionFeat) try { en.value.close() } catch (_: Throwable) {}
                iter.remove()
            }
        }

        // WICHTIG: erst laufen lassen, dann altes Ergebnis schließen – die Inputs
        // dieses Schritts sind die Tensoren des vorherigen Result-Objekts!
        val res = s.run(inputs)
        val old = lastResult
        lastResult = res
        try { old?.close() } catch (_: Throwable) {}

        val logitsT = res.get("logits").orElse(null) as? OnnxTensor
            ?: throw IllegalStateException("Output 'logits' fehlt")
        return readLastLogits(logitsT)
    }

    private fun lastResultHasCaches(): Boolean {
        val r = lastResult ?: return false
        if (cacheInputNames.isEmpty()) return false
        val first = cacheInputNames[0]
        val t = r.get(first).orElse(null) as? OnnxTensor ?: return false
        // Sequenzlänge > 0 => echtes present.* vorhanden
        return tensorShape(t).any { it > 0 } || cacheInputNames.any { n ->
            (r.get(n).orElse(null) as? OnnxTensor)?.let { tensorShape(it).any { d -> d > 0 } } == true
        }
    }

    /** Shape eines Tensors über dessen Metadaten (OnnxTensor hat kein getShape()). */
    private fun tensorShape(t: OnnxTensor): LongArray = try {
        (t.info as? ai.onnxruntime.TensorInfo)?.shape ?: LongArray(0)
    } catch (_: Throwable) {
        LongArray(0)
    }

    /** Liest die Logits der letzten Position aus [1,1,vocab] (oder [1,seq,vocab]). */
    private fun readLastLogits(t: OnnxTensor): FloatArray {
        val shape = tensorShape(t)
        val vocab = shape.last().toInt()
        val buf = t.floatBuffer
        val seqLen = if (shape.size >= 2) shape[shape.size - 2].toInt() else 1
        val start = (seqLen - 1) * vocab
        val out = FloatArray(vocab)
        buf.position(start)
        buf.get(out)
        return out
    }

    // ------------------------------------------------------------------ Sampling

    private fun sample(logits: FloatArray, cfg: GenConfig, recent: List<Int>): Int {
        // 1) Wiederholungsstrafe (Fenster)
        if (cfg.repPenalty != 1.0f && recent.isNotEmpty()) {
            val from = maxOf(0, recent.size - cfg.repWindow)
            val seen = HashSet<Int>(recent.subList(from, recent.size))
            for (id in seen) {
                if (id < 0 || id >= logits.size) continue
                val l = logits[id]
                logits[id] = if (l < 0f) l * cfg.repPenalty else l / cfg.repPenalty
            }
        }
        // 2) Greedy bei sehr niedriger Temperatur
        if (cfg.temperature <= 0.01f) {
            var best = 0
            for (i in 1 until logits.size) if (logits[i] > logits[best]) best = i
            return best
        }
        // 3) Top-k: k-th größter Wert via sortierter Kopie
        val n = logits.size
        var candIdx: IntArray
        var candVal: FloatArray
        if (cfg.topK in 1 until n) {
            val sorted = logits.copyOf()
            java.util.Arrays.sort(sorted)
            val kth = sorted[n - cfg.topK]
            var cnt = 0
            for (i in 0 until n) if (logits[i] >= kth) cnt++
            candIdx = IntArray(cnt)
            candVal = FloatArray(cnt)
            var j = 0
            for (i in 0 until n) if (logits[i] >= kth) { candIdx[j] = i; candVal[j] = logits[i]; j++ }
        } else {
            candIdx = IntArray(n) { it }
            candVal = logits.copyOf()
        }
        // Sortiere Kandidaten absteigend (kleine Liste -> Insertion Sort)
        for (i in 1 until candVal.size) {
            val v = candVal[i]; val id = candIdx[i]
            var j = i - 1
            while (j >= 0 && candVal[j] < v) { candVal[j + 1] = candVal[j]; candIdx[j + 1] = candIdx[j]; j-- }
            candVal[j + 1] = v; candIdx[j + 1] = id
        }
        // 4) Temperatur
        val maxL = candVal[0]
        val temp = cfg.temperature
        // 5) Top-p (nucleus)
        var sumExp = 0.0
        val probs = DoubleArray(candVal.size)
        for (i in candVal.indices) {
            val e = Math.exp(((candVal[i] - maxL) / temp).toDouble())
            probs[i] = e
            sumExp += e
        }
        var cum = 0.0
        var cutoff = candVal.size - 1
        for (i in candVal.indices) {
            cum += probs[i] / sumExp
            if (cum >= cfg.topP) { cutoff = i; break }
        }
        // Sampling über verbleibende Kandidaten (gewichtet)
        var mass = 0.0
        for (i in 0..cutoff) mass += probs[i]
        var r = Math.random() * mass
        for (i in 0..cutoff) {
            r -= probs[i]
            if (r <= 0) return candIdx[i]
        }
        return candIdx[0]
    }

    // ------------------------------------------------------------------ Vision

    /** Ergebnis des Vision-Encoders: Feature-Tensor + tatsächliche Bildgröße. */
    class VisionResult(val feat: OnnxTensor, val fw: Int, val fh: Int)

    /** Skaliert ein Bitmap auf die Vision-Eingabegröße (Vielfache von 32, 0.26–1.6 MP). */
    fun planVision(bitmap: android.graphics.Bitmap): IntArray {
        var w = bitmap.width
        var h = bitmap.height
        val minP = 262_144   // 0.26M
        val maxP = 1_638_400 // 1.6M
        val pixels = w * h
        if (pixels > maxP) {
            val scale = Math.sqrt(maxP.toDouble() / pixels)
            w = (w * scale).toInt(); h = (h * scale).toInt()
        } else if (pixels < minP) {
            val scale = Math.sqrt(minP.toDouble() / pixels)
            w = (w * scale).toInt(); h = (h * scale).toInt()
        }
        w = (w / 32) * 32; h = (h / 32) * 32
        if (w < 32) w = 32
        if (h < 32) h = 32
        return intArrayOf(w, h)
    }

    /** Anzahl Feature-Tokens für eine Encoder-Eingabegröße (Patch 16, Merge 2). */
    fun imageTokenCount(fw: Int, fh: Int): Int = (fw / 32) * (fh / 32)

    /**
     * Bild -> Vision-Encoder -> Feature-Tensor.
     * Skaliert auf Vielfache von 32, Pixelzahl zwischen 0.26M und 1.6M,
     * Normalisierung (x/255 - 0.5) / 0.5, CHW-Layout [1,3,H,W].
     */
    private fun runVision(bitmap: android.graphics.Bitmap): VisionResult {
        val vs = visionSession ?: throw IllegalStateException("Kein Vision-Encoder geladen")
        val inName = vs.inputNames.firstOrNull()
            ?: throw IllegalStateException("Vision-Encoder hat keine Inputs")
        val inInfo = vs.inputInfo[inName]?.info as? ai.onnxruntime.TensorInfo
            ?: throw IllegalStateException("Vision-Input-Metadaten fehlen")

        var w = bitmap.width
        var h = bitmap.height
        // Pixelzahl klemmen
        val minP = 262_144   // 0.26M
        val maxP = 1_638_400 // 1.6M
        val pixels = w * h
        if (pixels > maxP) {
            val scale = Math.sqrt(maxP.toDouble() / pixels)
            w = (w * scale).toInt(); h = (h * scale).toInt()
        } else if (pixels < minP) {
            val scale = Math.sqrt(minP.toDouble() / pixels)
            w = (w * scale).toInt(); h = (h * scale).toInt()
        }
        w = (w / 32) * 32; h = (h / 32) * 32
        if (w < 32) w = 32
        if (h < 32) h = 32

        val scaled = android.graphics.Bitmap.createScaledBitmap(bitmap, w, h, true)
        val px = IntArray(w * h)
        scaled.getPixels(px, 0, w, 0, 0, w, h)

        // Erwartete Input-Form: [1,3,H,W] (CHW), ggf. statische H/W aus Metadaten
        var tw = w.toLong(); var th = h.toLong()
        if (inInfo.shape != null && inInfo.shape.size == 4) {
            if (inInfo.shape[2] > 0) th = inInfo.shape[2]
            if (inInfo.shape[3] > 0) tw = inInfo.shape[3]
        }
        val fw = tw.toInt(); val fh = th.toInt()
        if (fw != w || fh != h) {
            val s2 = android.graphics.Bitmap.createScaledBitmap(scaled, fw, fh, true)
            s2.getPixels(px, 0, fw, 0, 0, fw, fh)
        }

        val data = FloatArray(3 * fh * fw)
        var iPtr = 0
        var idx = 0
        for (y in 0 until fh) {
            for (x in 0 until fw) {
                val p = px[idx++]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                data[iPtr] = (r - 0.5f) / 0.5f; iPtr++
                data[1 * fh * fw + y * fw + x] = (g - 0.5f) / 0.5f
                data[2 * fh * fw + y * fw + x] = (b - 0.5f) / 0.5f
            }
        }
        val shape = longArrayOf(1, 3, fh.toLong(), fw.toLong())
        val input = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        val res = vs.run(mapOf(inName to input))
        val out = res.get(0) as? OnnxTensor ?: throw IllegalStateException("Vision-Output fehlt")
        val feat = copyTensor(out)
        res.close()
        input.close()
        return VisionResult(feat, fw, fh)
    }

    /** Kopiert einen Tensor in einen eigenständigen fp32-Tensor (Result schließt sonst die Werte mit). */
    private fun copyTensor(t: OnnxTensor): OnnxTensor {
        val shape = tensorShape(t)
        val n = shape.fold(1L) { a, b -> a * b }.toInt()
        return when (t.type) {
            OnnxJavaType.FLOAT16 -> {
                val sb = t.shortBuffer
                val f = FloatArray(n)
                for (i in 0 until n) f[i] = halfToFloat(sb.get(i))
                OnnxTensor.createTensor(env, FloatBuffer.wrap(f), shape)
            }
            else -> {
                val fb = t.floatBuffer
                val f = FloatArray(n)
                fb.get(f)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(f), shape)
            }
        }
    }

    // --------------------------------------------------- d1 System-One-Decide

    /** float16 <-> float (Bit-Konvertierung). */
    private fun halfToFloat(h: Short): Float {
        val bits = h.toInt() and 0xFFFF
        val sign = (bits and 0x8000) shl 16
        val expBits = (bits and 0x7C00) shr 10
        val mant = bits and 0x03FF
        val out = when (expBits) {
            0 -> if (mant == 0) sign else {
                var e = -1
                var m = mant
                while (m and 0x0400 == 0) { m = m shl 1; e-- }
                m = m and 0x03FF
                sign or ((127 - 15 + e + 1) shl 23) or (m shl 13)
            }
            0x1F -> sign or 0x7F800000 or (mant shl 13)
            else -> sign or ((expBits - 15 + 127) shl 23) or (mant shl 13)
        }
        return Float.fromBits(out)
    }

    private fun floatToHalf(f: Float): Short {
        val bits = java.lang.Float.floatToIntBits(f)
        val sign = (bits shr 16) and 0x8000
        val expBits = (bits shr 23) and 0xFF
        var mant = bits and 0x007FFFFF
        if (expBits == 0xFF) return ((sign or 0x7C00) or (if (mant != 0) 1 else 0)).toShort()
        var e = expBits - 127 + 15
        if (e >= 0x1F) return (sign or 0x7C00).toShort()
        if (e <= 0) {
            if (e < -10) return sign.toShort()
            mant = mant or 0x00800000
            val shift = 14 - e
            val half = (mant shr shift) or (if ((mant shr (shift - 1)) and 1 == 1) 1 else 0)
            return (sign or half).toShort()
        }
        return (sign or (e shl 10) or (mant shr 13)).toShort()
    }

    private fun inputRank(name: String): Int = try {
        (session?.inputInfo?.get(name)?.info as? ai.onnxruntime.TensorInfo)?.shape?.size ?: 2
    } catch (_: Throwable) { 2 }

    private fun inputType(name: String): OnnxJavaType = try {
        (session?.inputInfo?.get(name)?.info as? ai.onnxruntime.TensorInfo)?.type ?: OnnxJavaType.INT64
    } catch (_: Throwable) { OnnxJavaType.INT64 }

    private fun feedIntTensor(inputs: HashMap<String, OnnxTensor>, name: String, data: IntArray) {
        val rank = inputRank(name)
        val type = inputType(name)
        val t = if (rank <= 1) {
            when (type) {
                OnnxJavaType.INT32 -> OnnxTensor.createTensor(env, IntBuffer.wrap(data), longArrayOf(data.size.toLong()))
                else -> {
                    val l = LongArray(data.size) { data[it].toLong() }
                    OnnxTensor.createTensor(env, LongBuffer.wrap(l), longArrayOf(data.size.toLong()))
                }
            }
        } else {
            when (type) {
                OnnxJavaType.INT32 -> OnnxTensor.createTensor(env, IntBuffer.wrap(data), longArrayOf(1, data.size.toLong()))
                else -> {
                    val l = LongArray(data.size) { data[it].toLong() }
                    OnnxTensor.createTensor(env, LongBuffer.wrap(l), longArrayOf(1, data.size.toLong()))
                }
            }
        }
        inputs[name] = t
    }

    /**
     * Ein System-One-Decision: für jeden Pass (State + bis zu 16 Fragen, gepackt)
     * ein Forward-Pass; die Bilder werden einmal encodiert. Rückgabe: die Logits
     * an den Antwort-Slots, in Frage-Reihenfolge (über alle Pässe fortlaufend).
     */
    fun decide(passes: List<D1.D1Pass>, images: List<android.graphics.Bitmap>): Array<FloatArray> {
        val s = session ?: throw IllegalStateException("Kein Modell geladen")
        var feat: OnnxTensor? = null
        var featOwned = false
        try {
            if (images.isNotEmpty()) {
                if (visionSession == null || visionInputName == null) {
                    throw IllegalStateException("Kein Vision-Encoder geladen (Vision-Modell fehlt)")
                }
                val results = images.map { runVision(it) }
                try {
                    if (results.size == 1) {
                        feat = adaptVisionToDecoder(results[0].feat)
                        featOwned = feat !== results[0].feat
                    } else {
                        feat = concatVision(results)
                        featOwned = true
                    }
                } finally {
                    for (r in results) if (r.feat !== feat) try { r.feat.close() } catch (_: Throwable) {}
                }
            }
            val all = ArrayList<FloatArray>(64)
            for (pass in passes) {
                val logits = runDecidePass(s, pass, feat)
                if (logits.size != pass.slotCount) {
                    throw IllegalStateException("Graph lieferte ${logits.size} Slot-Logits, erwartet ${pass.slotCount}")
                }
                all.addAll(logits)
            }
            return all.toTypedArray()
        } finally {
            if (featOwned) try { feat?.close() } catch (_: Throwable) {}
        }
    }

    /** Bringt Vision-Features auf den Dtype/Rang, den der Decoder-Input erwartet. */
    private fun adaptVisionToDecoder(src: OnnxTensor): OnnxTensor {
        val vn = visionInputName ?: return src
        val info = try { session?.inputInfo?.get(vn)?.info as? ai.onnxruntime.TensorInfo } catch (_: Throwable) { null }
        val wantType = info?.type ?: OnnxJavaType.FLOAT
        val wantRank = info?.shape?.size ?: src.info.let { (it as? ai.onnxruntime.TensorInfo)?.shape?.size ?: 2 }
        val srcShape = (src.info as? ai.onnxruntime.TensorInfo)?.shape ?: LongArray(0)
        val srcRank = srcShape.size
        if (wantType == src.type && wantRank == srcRank) return src
        // Werte kopieren
        val d = srcShape.last()
        val n = srcShape.dropLast(1).fold(1L) { a, b -> a * b }
        val floats = FloatArray((n * d).toInt())
        when (src.type) {
            OnnxJavaType.FLOAT16 -> {
                val sb = src.shortBuffer
                for (i in floats.indices) floats[i] = halfToFloat(sb.get(i))
            }
            else -> {
                val fb = src.floatBuffer
                fb.get(floats)
            }
        }
        val shape = if (wantRank >= 3) longArrayOf(1, n, d)
            else if (wantRank == 2) longArrayOf(n, d)
            else longArrayOf(n * d)
        return when (wantType) {
            OnnxJavaType.FLOAT16 -> {
                val shorts = ShortArray(floats.size) { floatToHalf(floats[it]) }
                OnnxTensor.createTensor(env, ShortBuffer.wrap(shorts), shape, OnnxJavaType.FLOAT16)
            }
            else -> OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), shape)
        }
    }

    /** Mehrere Bilder: Features an der Token-Achse verketten [1, N1+N2, D]. */
    private fun concatVision(results: List<VisionResult>): OnnxTensor {
        val shape0 = (results[0].feat.info as? ai.onnxruntime.TensorInfo)?.shape ?: LongArray(0)
        val d = shape0.last()
        val nTotal = results.sumOf { r ->
            val sh = (r.feat.info as? ai.onnxruntime.TensorInfo)?.shape ?: LongArray(0)
            sh.dropLast(1).fold(1L) { a, b -> a * b }
        }
        val floats = FloatArray((nTotal * d).toInt())
        var at = 0
        for (r in results) {
            val sh = (r.feat.info as? ai.onnxruntime.TensorInfo)?.shape ?: LongArray(0)
            val n = sh.dropLast(1).fold(1L) { a, b -> a * b }.toInt()
            val src = adaptVisionToDecoder(r.feat)
            when (src.type) {
                OnnxJavaType.FLOAT16 -> {
                    val sb = src.shortBuffer
                    for (i in 0 until n * d.toInt()) floats[at + i] = halfToFloat(sb.get(i))
                }
                else -> {
                    val fb = src.floatBuffer.duplicate()
                    fb.get(floats, at, n * d.toInt())
                }
            }
            if (src !== r.feat) try { src.close() } catch (_: Throwable) {}
            at += n * d.toInt()
        }
        val shape = longArrayOf(1, nTotal, d)
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), shape)
    }

    /** Ein Forward-Pass für eine gepackte d1-Zeile; Rückgabe: Slot-Logits [S][vocab]. */
    private fun runDecidePass(s: OrtSession, pass: D1.D1Pass, feat: OnnxTensor?): List<FloatArray> {
        val inputs = HashMap<String, OnnxTensor>()
        inputs[inputIdsName] = OnnxTensor.createTensor(env,
            LongBuffer.wrap(LongArray(pass.inputIds.size) { pass.inputIds[it].toLong() }),
            longArrayOf(1, pass.inputIds.size.toLong()))
        val mask = LongArray(pass.inputIds.size) { 1L }
        inputs[attnName] = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), longArrayOf(1, mask.size.toLong()))
        if (hasPosInput) {
            val pos = LongArray(pass.inputIds.size) { it.toLong() }
            inputs[posName] = OnnxTensor.createTensor(env, LongBuffer.wrap(pos), longArrayOf(1, pos.size.toLong()))
        }
        feedIntTensor(inputs, "question_ids", pass.questionIds)
        feedIntTensor(inputs, "slot_positions", pass.slotPositions)
        if (feat != null && visionInputName != null && visionInputName in sessionInputs) {
            inputs[visionInputName!!] = feat
        }

        // Sicherheitsnetz: unbekannte Inputs nicht füttern
        val iter = inputs.entries.iterator()
        while (iter.hasNext()) {
            val en = iter.next()
            if (en.key !in sessionInputs) {
                try { en.value.close() } catch (_: Throwable) {}
                iter.remove()
            }
        }

        val res = s.run(inputs)
        try {
            val logitsT = res.get("logits").orElse(null) as? OnnxTensor
                ?: throw IllegalStateException("Output 'logits' fehlt")
            return readSlotLogits(logitsT, pass.slotPositions.size)
        } finally {
            try { res.close() } catch (_: Throwable) {}
        }
    }

    /** Logits an den Slots [S][vocab] – verträgt [S,V], [1,S,V] und fp16. */
    private fun readSlotLogits(t: OnnxTensor, expectedSlots: Int): List<FloatArray> {
        val shape = tensorShape(t)
        val vocab = shape.last().toInt()
        val slots = if (shape.size >= 2) shape[shape.size - 2].toInt() else 1
        if (slots != expectedSlots && !(slots == 1 && expectedSlots == 1)) {
            throw IllegalStateException("Slot-Logits: ${slots} Zeilen, ${expectedSlots} erwartet")
        }
        val n = slots * vocab
        val out = ArrayList<FloatArray>(slots)
        when (t.type) {
            OnnxJavaType.FLOAT16 -> {
                val sb = t.shortBuffer
                for (row in 0 until slots) {
                    val f = FloatArray(vocab)
                    for (i in 0 until vocab) f[i] = halfToFloat(sb.get(row * vocab + i))
                    out.add(f)
                }
            }
            else -> {
                val fb = t.floatBuffer
                for (row in 0 until slots) {
                    val f = FloatArray(vocab)
                    fb.get(f)
                    out.add(f)
                }
            }
        }
        return out
    }

    private fun closeVisionFeat() {
        try { visionFeat?.close() } catch (_: Throwable) {}
        visionFeat = null
    }

    private fun closeLastResult() {
        try { lastResult?.close() } catch (_: Throwable) {}
        lastResult = null
    }

    /** Setzt den Cache-Zustand zurück (vor jeder neuen Generierung). */
    fun reset() {
        closeLastResult()
        closeVisionFeat()
    }

    // ------------------------------------------------------------------ Aufräumen

    fun close() {
        closeLastResult()
        closeVisionFeat()
        zeroCaches?.let { zc -> for (t in zc.values) try { t.close() } catch (_: Throwable) {} }
        zeroCaches = null
        try { visionSession?.close() } catch (_: Throwable) {}
        visionSession = null
        try { session?.close() } catch (_: Throwable) {}
        session = null
    }

    protected fun finalize() {
        close()
    }
}
