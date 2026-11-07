package ai.liquid.lfm3.ui

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import ai.liquid.lfm3.R
import ai.liquid.lfm3.engine.EngineHost
import ai.liquid.lfm3.engine.ModelCatalog
import ai.liquid.lfm3.engine.ModelInfo
import ai.liquid.lfm3.engine.ModelManager
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Modell-Verwaltung: alle LFM3-Varianten mit allen Präzisionen,
 * Download (mit Fortschritt/Abbruch), Laden in die Engine, Löschen (Langer Druck).
 *
 * Lifecycle-Regeln (Fix für "Fragment not attached to a context"):
 * - Download-/Lade-Zustand liegt im companion object und überlebt Tab-Wechsel.
 * - Kein requireContext() in Callbacks, die nach onDetach laufen können.
 * - UI-Updates gehen über activeAdapter (der aktuell sichtbaren Liste).
 */
class ModelsFragment : Fragment(R.layout.fragment_models), ModelsAdapter.Listener {

    private class DownloadState {
        val cancel = AtomicBoolean(false)
        @Volatile var pct: Int = 0
        @Volatile var done: Long = 0
        @Volatile var total: Long = 1
    }

    companion object {
        // Überlebt Fragment-Wechsel: Downloads laufen im Application-Kontext weiter;
        // beim Zurückkehren in den Tab wird der laufende Stand wieder angezeigt.
        private val downloads = ConcurrentHashMap<String, DownloadState>()
        @Volatile private var loadingModelId: String? = null
        private val dlExecutor = Executors.newSingleThreadExecutor { Thread(it, "lfm3-download") }
        private val main = Handler(Looper.getMainLooper())
        @Volatile private var activeAdapter: ModelsAdapter? = null

        private fun posOf(m: ModelInfo): Int = ModelCatalog.MODELS.indexOfFirst { it.id == m.id }
    }

    private lateinit var adapter: ModelsAdapter
    private lateinit var tvStorage: TextView

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        EngineHost.init(requireContext())

        tvStorage = view.findViewById(R.id.tv_storage)
        adapter = ModelsAdapter(ModelCatalog.MODELS, this)
        val rv = view.findViewById<RecyclerView>(R.id.rv_models)
        rv.layoutManager = LinearLayoutManager(requireContext())
        rv.adapter = adapter
        activeAdapter = adapter

        refreshStorage()
    }

    private fun refreshStorage() {
        // CRASH-FIX: früher requireContext() -> IllegalStateException nach Tab-Wechsel
        val ctx = context ?: return
        val used = ModelManager.diskUsedBytes(ctx)
        val free = ModelManager.diskFreeBytes(ctx)
        if (this::tvStorage.isInitialized) {
            tvStorage.text = "Belegt: ${ModelCatalog.humanBytes(used)} · Frei: ${ModelCatalog.humanBytes(free)}"
        }
    }

    private fun key(m: ModelInfo, p: String) = "${m.id}|$p"

    // ---------------------------------------------------------------- Listener-Implementierung

    override fun onCyclePrecision(model: ModelInfo, current: String): String {
        val ids = model.precisions.map { it.id }
        val idx = ids.indexOf(current)
        return ids[(idx + 1).mod(ids.size)]
    }

    override fun onAction(model: ModelInfo, precId: String) {
        val k = key(model, precId)
        val prec = model.precision(precId) ?: return
        val ds = downloads[k]
        if (ds != null) {
            ds.cancel.set(true)
            logLine("Download abgebrochen: ${model.displayName} $precId")
            return
        }
        if (loadingModelId != null) {
            toast("Es wird bereits ein Modell geladen")
            return
        }
        if (ModelManager.isInstalled(requireContext(), model, prec)) {
            load(model, prec)
        } else {
            startDownload(model, prec)
        }
    }

    override fun onDelete(model: ModelInfo, precId: String) {
        val prec = model.precision(precId) ?: return
        // Application-Kontext für den Dialog-Callback cachen: Das Fragment kann
        // detached sein, wenn der User bei offenem Dialog den Tab wechselt.
        val appCtx = requireContext().applicationContext
        androidx.appcompat.app.AlertDialog.Builder(requireContext())
            .setTitle("Löschen")
            .setMessage("${model.displayName} [$precId] wirklich löschen?")
            .setPositiveButton(R.string.delete) { _, _ ->
                ModelManager.delete(appCtx, model, prec)
                toast("Gelöscht: ${model.displayName} $precId")
                activeAdapter?.notifyDataSetChanged()
                refreshStorage()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun startDownload(model: ModelInfo, prec: ai.liquid.lfm3.engine.PrecisionInfo) {
        val need = model.totalBytes(prec.id)
        val free = ModelManager.diskFreeBytes(requireContext())
        if (free in 1..(need + 200_000_000L)) {
            toast("Zu wenig Speicher: ${ModelCatalog.humanBytes(need)} nötig, ${ModelCatalog.humanBytes(free)} frei")
            return
        }
        val k = key(model, prec.id)
        val ds = DownloadState()
        downloads[k] = ds
        logLine("Download startet: ${model.displayName} [${prec.id}] (${ModelCatalog.humanBytes(need)})")
        activeAdapter?.notifyItemChanged(posOf(model))

        val ctx = requireContext().applicationContext
        dlExecutor.submit {
            var lastUi = 0L
            try {
                ModelManager.download(ctx, model, prec, ds.cancel) { p ->
                    val total = p.totalSize.coerceAtLeast(1)
                    ds.done = p.totalDone
                    ds.total = total
                    ds.pct = ((p.totalDone * 100) / total).toInt().coerceIn(0, 100)
                    // Nur alle 250 ms UI anfassen und NUR das eine Item neu binden.
                    val now = System.currentTimeMillis()
                    if (now - lastUi > 250) {
                        lastUi = now
                        main.post { activeAdapter?.notifyItemChanged(posOf(model)) }
                    }
                }
                main.post {
                    downloads.remove(k)
                    logLine("Fertig: ${model.displayName} [${prec.id}]")
                    activeAdapter?.notifyItemChanged(posOf(model))
                    refreshStorage()
                    toast("Modell heruntergeladen – kann jetzt geladen werden")
                }
            } catch (t: Throwable) {
                main.post {
                    downloads.remove(k)
                    val msg = if (t is InterruptedException) "abgebrochen" else "Fehler: ${t.message}"
                    logLine("Download ${model.displayName} [${prec.id}] $msg")
                    activeAdapter?.notifyItemChanged(posOf(model))
                    refreshStorage()
                }
            }
        }
    }

    private fun load(model: ModelInfo, prec: ai.liquid.lfm3.engine.PrecisionInfo) {
        loadingModelId = model.id
        activeAdapter?.notifyItemChanged(posOf(model))
        val ts = SimpleDateFormat("HH:mm:ss", Locale.GERMANY).format(Date())
        EngineHost.loadModel(requireContext().applicationContext, model, prec,
            log = { line ->
                main.post { logLine("[$ts] $line") }
            },
            onDone = { err ->
                main.post {
                    loadingModelId = null
                    activeAdapter?.notifyItemChanged(posOf(model))
                    if (err == null) {
                        toast("Geladen: ${model.displayName} · Backend: ${EngineHost.backendLabel}")
                    } else {
                        toast("Ladefehler: $err")
                    }
                    refreshStorage()
                }
            })
    }

    override fun stateText(model: ModelInfo, precId: String): String {
        val prec = model.precision(precId) ?: return ""
        val ctx = context ?: return ""
        val k = key(model, precId)
        return when {
            downloads.containsKey(k) -> getString(R.string.downloading)
            loadingModelId == model.id -> "wird geladen…"
            EngineHost.currentModel?.id == model.id && EngineHost.currentPrecision == precId ->
                getString(R.string.installed) + " · AKTIV (${EngineHost.backendLabel})"
            ModelManager.isInstalled(ctx, model, prec) -> getString(R.string.installed)
            else -> "nicht geladen"
        }
    }

    override fun progress(model: ModelInfo, precId: String): Triple<Int, Long, Long>? {
        val ds = downloads[key(model, precId)] ?: return null
        return Triple(ds.pct, ds.done, ds.total)
    }

    override fun isBusy(model: ModelInfo): Boolean =
        downloads.containsKey(key(model, adapter.precisionOf(model))) || loadingModelId == model.id

    // ---------------------------------------------------------------- Helpers

    private fun logLine(line: String) {
        android.util.Log.i("LFM3", line)
        // View-Zugriff nur, wenn die View existiert (Fragment kann detached sein)
        if (!this::tvStorage.isInitialized) return
        val cur = tvStorage.text.toString()
        tvStorage.text = line
        tvStorage.postDelayed({
            if (tvStorage.text == line) refreshStorage()
        }, 4000)
        if (cur.isNotEmpty() && cur != line && !cur.startsWith("Belegt:")) {
            // vorherige Zeile verwerfen (bewusst minimal)
        }
    }

    private fun toast(msg: String) {
        context?.let { Toast.makeText(it, msg, Toast.LENGTH_LONG).show() }
    }

    override fun onDestroyView() {
        if (activeAdapter === adapter) activeAdapter = null
        super.onDestroyView()
        // Downloads laufen bewusst weiter (Application-Kontext + companion object)
    }
}
