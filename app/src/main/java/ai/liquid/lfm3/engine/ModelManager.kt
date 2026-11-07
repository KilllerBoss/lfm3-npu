package ai.liquid.lfm3.engine

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Verwaltet Modell-Downloads von HuggingFace (Progress, Abbruch, Löschen,
 * Installationsprüfung). Dateien liegen unter
 * <external-files-dir>/models/{modelId}/{relativerPfad}.
 */
object ModelManager {

    data class Progress(val file: String, val fileDone: Long, val fileSize: Long,
                        val totalDone: Long, val totalSize: Long)

    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    fun modelsRoot(ctx: Context): File =
        File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "models")

    fun modelDir(ctx: Context, modelId: String): File = File(modelsRoot(ctx), modelId)

    fun diskFreeBytes(ctx: Context): Long = try {
        (ctx.getExternalFilesDir(null) ?: ctx.filesDir).usableSpace
    } catch (_: Throwable) { 0L }

    fun diskUsedBytes(ctx: Context): Long = try {
        var s = 0L
        modelsRoot(ctx).walkTopDown().filter { it.isFile }.forEach { s += it.length() }
        s
    } catch (_: Throwable) { 0L }

    /** Prüft, ob alle Dateien der Präzision (inkl. Common) vollständig vorhanden sind. */
    fun isInstalled(ctx: Context, model: ModelInfo, prec: PrecisionInfo): Boolean {
        val all: List<FileSpec> = model.commonFiles + prec.files
        for (f in all) {
            val file = File(modelDir(ctx, model.id), f.path)
            if (!file.isFile) return false
            if (f.bytes > 0 && file.length() < f.bytes * 99L / 100L) return false
        }
        return true
    }

    fun installedBytes(ctx: Context, model: ModelInfo, prec: PrecisionInfo): Long {
        val all: List<FileSpec> = model.commonFiles + prec.files
        var s = 0L
        for (f in all) {
            val file = File(modelDir(ctx, model.id), f.path)
            if (file.isFile) s += file.length()
        }
        return s
    }

    fun delete(ctx: Context, model: ModelInfo, prec: PrecisionInfo) {
        val dir = modelDir(ctx, model.id)
        val all: List<FileSpec> = model.commonFiles + prec.files
        for (f in all) {
            File(dir, f.path).delete()
            File(dir, f.path + ".part").delete()
        }
        // Leere onnx-Verzeichnisse aufräumen
        try { File(dir, "onnx").deleteRecursively() } catch (_: Throwable) {}
        try {
            if (dir.isDirectory && dir.list()?.isEmpty() == true) dir.delete()
        } catch (_: Throwable) {}
    }

    /**
     * Lädt alle Dateien einer Präzision herunter (blockierend – im Worker-Thread aufrufen).
     * Bereits vollständig vorhandene Dateien werden übersprungen (Resume-freundlich).
     */
    fun download(
        ctx: Context,
        model: ModelInfo,
        prec: PrecisionInfo,
        cancel: AtomicBoolean,
        onProgress: (Progress) -> Unit
    ) {
        val all: List<FileSpec> = model.commonFiles + prec.files
        val total = all.sumOf { it.bytes }.coerceAtLeast(1L)
        var done = 0L

        for (f in all) {
            throwIfCancelled(cancel)
            val target = File(modelDir(ctx, model.id), f.path)
            target.parentFile?.mkdirs()
            val part = File(target.parentFile, target.name + ".part")

            if (target.isFile && (f.bytes <= 0 || target.length() >= f.bytes * 99L / 100L)) {
                done += f.bytes
                onProgress(Progress(f.path, f.bytes, f.bytes, done, total))
                continue
            }
            // Ohne Range-Support: angebrochene Dateien neu beginnen
            if (part.exists()) part.delete()

            val url = "https://huggingface.co/${model.repo}/resolve/main/${f.path}"
            val req = Request.Builder()
                .url(url)
                .header("User-Agent", "LFM3-NPU-App/1.0 (Android)")
                .build()

            client.newCall(req).execute().use { resp ->
                if (!resp.isSuccessful) {
                    // Optionale Dateien (chat_template.jinja) dürfen 404 sein
                    if (resp.code == 404 && isOptional(f.path)) {
                        done += f.bytes
                        onProgress(Progress(f.path, f.bytes, f.bytes, done, total))
                        return@use
                    }
                    throw IOException("HTTP ${resp.code} für $f.path")
                }
                val body = resp.body ?: throw IOException("Leere Antwort für ${f.path}")
                val contentLen = body.contentLength()
                var localDone = 0L
                body.byteStream().use { input ->
                    java.io.FileOutputStream(part, false).use { out ->
                        val buf = ByteArray(256 * 1024)
                        var lastUi = 0L
                        while (true) {
                            throwIfCancelled(cancel)
                            val r = input.read(buf)
                            if (r < 0) break
                            out.write(buf, 0, r)
                            localDone += r
                            done += r
                            val now = System.currentTimeMillis()
                            if (now - lastUi > 200) {
                                lastUi = now
                                onProgress(Progress(f.path, localDone,
                                    if (contentLen > 0) contentLen else f.bytes, done, total))
                            }
                        }
                        out.flush()
                    }
                }
                // Größenprüfung
                if (f.bytes > 0 && part.length() < f.bytes * 99L / 100L) {
                    throw IOException("Unvollständig: ${f.path} (${part.length()}/${f.bytes})")
                }
                if (target.exists()) target.delete()
                if (!part.renameTo(target)) {
                    throw IOException("Umbenennen fehlgeschlagen: ${part.name}")
                }
            }
            onProgress(Progress(f.path, f.bytes, f.bytes, done, total))
        }
    }

    private fun isOptional(path: String): Boolean =
        path.endsWith("chat_template.jinja") || path.endsWith(".md")

    private fun throwIfCancelled(cancel: AtomicBoolean) {
        if (cancel.get()) throw InterruptedException("Download abgebrochen")
    }
}
