package ai.liquid.lfm3.engine

import android.content.Context
import android.os.Handler
import android.os.Looper
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Zentrale Verwaltung von Downloads + Laden/Entladen für den Launcher.
 * Zustand liegt hier (Singleton) und überlebt Activity-Neuaufbau;
 * die UI meldet sich per [listener] und liest die States on demand.
 */
object ModelTasks {

    enum class State { NOT_INSTALLED, DOWNLOADING, INSTALLED, LOADING, LOADED }

    class DownloadState(
        val cancel: AtomicBoolean = AtomicBoolean(false),
        @Volatile var pct: Int = 0,
        @Volatile var done: Long = 0,
        @Volatile var total: Long = 1
    )

    private val downloads = ConcurrentHashMap<String, DownloadState>()
    private val main = Handler(Looper.getMainLooper())
    private val dlExecutor = Executors.newSingleThreadExecutor { Thread(it, "lfm3-download") }

    @Volatile var loadingModelId: String? = null
        private set
    @Volatile var lastMessage: String? = null
        private set

    /** UI-Hook: wird auf dem Main-Thread aufgerufen, wenn sich etwas geändert hat. */
    @Volatile var listener: (() -> Unit)? = null

    private fun notifyUi() = main.post { try { listener?.invoke() } catch (_: Throwable) {} }

    // ------------------------------------------------------------- Präzision

    fun selectedPrecision(ctx: Context, m: ModelInfo): String {
        val prefs = ctx.getSharedPreferences("lfm3", Context.MODE_PRIVATE)
        val saved = prefs.getString("prec_${m.id}", ModelCatalog.DEFAULT_PRECISION)
            ?: ModelCatalog.DEFAULT_PRECISION
        return if (m.precision(saved) != null) saved else ModelCatalog.DEFAULT_PRECISION
    }

    fun cyclePrecision(ctx: Context, m: ModelInfo): String {
        val ids = m.precisions.map { it.id }
        val cur = selectedPrecision(ctx, m)
        val next = ids[(ids.indexOf(cur) + 1).mod(ids.size)]
        ctx.getSharedPreferences("lfm3", Context.MODE_PRIVATE)
            .edit().putString("prec_${m.id}", next).apply()
        return next
    }

    fun precisionInfo(ctx: Context, m: ModelInfo): PrecisionInfo =
        m.precision(selectedPrecision(ctx, m)) ?: m.precisions.first()

    // ------------------------------------------------------------- Zustände

    fun stateOf(ctx: Context, m: ModelInfo): State {
        val prec = precisionInfo(ctx, m)
        return when {
            downloads.containsKey("${m.id}|${prec.id}") -> State.DOWNLOADING
            loadingModelId == m.id -> State.LOADING
            EngineHost.currentModel?.id == m.id -> State.LOADED
            ModelManager.isInstalled(ctx, m, prec) -> State.INSTALLED
            else -> State.NOT_INSTALLED
        }
    }

    fun downloadState(m: ModelInfo, ctx: Context): DownloadState? =
        downloads["${m.id}|${precisionInfo(ctx, m).id}"]

    // ------------------------------------------------------------- Aktionen

    /** Tippen auf eine Modell-Zeile. Rückgabe: Meldung für den User (oder null). */
    fun onRowTap(ctx: Context, m: ModelInfo): String? {
        val prec = precisionInfo(ctx, m)
        val state = stateOf(ctx, m)
        return when (state) {
            State.LOADING -> "Läuft bereits: ${m.displayName}"
            State.DOWNLOADING -> {
                downloads["${m.id}|${prec.id}"]?.cancel?.set(true)
                "Download abgebrochen: ${m.displayName}"
            }
            State.LOADED -> {
                EngineHost.unloadModel()
                "Entladen: ${m.displayName}"
            }
            State.INSTALLED -> { load(ctx, m, prec); null }
            State.NOT_INSTALLED -> {
                val need = m.totalBytes(prec.id)
                val free = ModelManager.diskFreeBytes(ctx)
                if (free in 1..(need + 200_000_000L)) {
                    "Zu wenig Speicher: ${ModelCatalog.humanBytes(need)} nötig, ${ModelCatalog.humanBytes(free)} frei"
                } else {
                    download(ctx, m, prec)
                    null
                }
            }
        }
    }

    fun download(ctx: Context, m: ModelInfo, prec: PrecisionInfo) {
        val k = "${m.id}|${prec.id}"
        if (downloads.containsKey(k)) return
        val ds = DownloadState()
        downloads[k] = ds
        lastMessage = "Download startet: ${m.displayName} [${prec.id}] (${ModelCatalog.humanBytes(m.totalBytes(prec.id))})"
        notifyUi()

        val appCtx = ctx.applicationContext
        dlExecutor.submit {
            var lastUi = 0L
            try {
                ModelManager.download(appCtx, m, prec, ds.cancel) { p ->
                    val total = p.totalSize.coerceAtLeast(1)
                    ds.done = p.totalDone
                    ds.total = total
                    ds.pct = ((p.totalDone * 100) / total).toInt().coerceIn(0, 100)
                    val now = System.currentTimeMillis()
                    if (now - lastUi > 250) {
                        lastUi = now
                        notifyUi()
                    }
                }
                downloads.remove(k)
                lastMessage = "Fertig geladen: ${m.displayName} [${prec.id}] – starte Modell …"
                notifyUi()
                // Komfort: nach dem Download automatisch laden
                load(appCtx, m, prec)
            } catch (t: Throwable) {
                downloads.remove(k)
                lastMessage = if (t is InterruptedException) "Download abgebrochen: ${m.displayName}"
                else "Download-Fehler ${m.displayName}: ${t.message}"
                notifyUi()
            }
        }
    }

    /** Lädt ein Modell in die Engine. onDone läuft im Engine-Thread. */
    fun load(ctx: Context, m: ModelInfo, prec: PrecisionInfo, onDone: ((String?) -> Unit)? = null) {
        if (loadingModelId != null) { onDone?.invoke("Es wird bereits ein Modell geladen"); return }
        loadingModelId = m.id
        notifyUi()
        EngineHost.loadModel(ctx.applicationContext, m, prec,
            log = { /* Logcat */ },
            onDone = { err ->
                loadingModelId = null
                lastMessage = if (err == null) "Geladen: ${m.displayName} · ${EngineHost.backendLabel}"
                else "Ladefehler ${m.displayName}: $err"
                notifyUi()
                onDone?.invoke(err)
            })
    }

    fun delete(ctx: Context, m: ModelInfo) {
        val prec = precisionInfo(ctx, m)
        if (EngineHost.currentModel?.id == m.id) EngineHost.unloadModel()
        ModelManager.delete(ctx.applicationContext, m, prec)
        lastMessage = "Gelöscht: ${m.displayName} [${prec.id}]"
        notifyUi()
    }
}
