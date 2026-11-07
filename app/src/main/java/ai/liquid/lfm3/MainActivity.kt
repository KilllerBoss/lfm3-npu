package ai.liquid.lfm3

import android.annotation.SuppressLint
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.text.format.Formatter
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.core.widget.doAfterTextChanged
import ai.liquid.lfm3.engine.EngineHost
import ai.liquid.lfm3.engine.ModelCatalog
import ai.liquid.lfm3.engine.ModelInfo
import ai.liquid.lfm3.engine.ModelTasks
import ai.liquid.lfm3.server.ServerHost
import ai.liquid.lfm3.ui.DemoHtml
import ai.liquid.lfm3.ui.HtmlRunnerActivity
import ai.liquid.lfm3.ui.SysPrompt
import java.io.File

/**
 * HTML-Launcher: lokale Server-App, deren HTML-Apps (von einer externen KI
 * geschrieben) das on-device LFM3-Modell über die lokale HTTP-API nutzen.
 *
 * Ablauf: Systemprompt kopieren -> externe KI schreibt HTML -> Code hier
 * einfügen -> "Starten" -> HTML läuft fullscreen im WebView.
 */
class MainActivity : AppCompatActivity() {

    private class RowViews(val root: View, val name: TextView, val sub: TextView,
                           val state: TextView, val progress: ProgressBar)

    private lateinit var tvServer: TextView
    private lateinit var statusDot: View
    private lateinit var modelPanel: LinearLayout
    private lateinit var modelList: LinearLayout
    private lateinit var etHtml: EditText
    private lateinit var tvHtmlInfo: TextView
    private val rows = HashMap<String, RowViews>()

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            EngineHost.init(applicationContext)
            ServerHost.start(applicationContext)
            setContentView(R.layout.activity_main)

            tvServer = findViewById(R.id.tv_server)
            statusDot = findViewById(R.id.status_dot)
            modelPanel = findViewById(R.id.model_panel)
            modelList = findViewById(R.id.model_list)
            etHtml = findViewById(R.id.et_html)
            tvHtmlInfo = findViewById(R.id.tv_htmlinfo)

            // Edge-to-Edge (targetSdk 35): Statusleiste oben, Tastatur unten
            val root = findViewById<ViewGroup>(R.id.root)
            ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
                val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
                val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
                v.updatePadding(top = bars.top)
                v.updatePadding(bottom = maxOf(bars.bottom, ime.bottom))
                WindowInsetsCompat.CONSUMED
            }

            // Zuletzt genutztes HTML wiederherstellen
            if (etHtml.text.isNullOrBlank()) {
                val f = lastHtmlFile()
                if (f.isFile) try { etHtml.setText(f.readText()) } catch (_: Throwable) {}
            }
            etHtml.doAfterTextChanged { updateHtmlInfo() }
            updateHtmlInfo()

            findViewById<Button>(R.id.btn_sysprompt).setOnClickListener { showSysPrompt() }
            findViewById<Button>(R.id.btn_models).setOnClickListener {
                modelPanel.visibility =
                    if (modelPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
                refreshAll()
            }
            findViewById<Button>(R.id.btn_example).setOnClickListener {
                etHtml.setText(DemoHtml.html)
                toast("Beispiel-App eingefügt – direkt starten oder anpassen")
            }
            findViewById<Button>(R.id.btn_clear).setOnClickListener { etHtml.setText("") }
            findViewById<Button>(R.id.btn_start).setOnClickListener { onStartClicked() }

            buildModelRows()
            ModelTasks.listener = { if (!isDestroyed && !isFinishing) refreshAll() }
            refreshAll()

            App.lastCrash?.let { showCrashBanner(it) }
        } catch (t: Throwable) {
            showCrashScreen(t)
        }
    }

    override fun onDestroy() {
        if (ModelTasks.listener !== null) ModelTasks.listener = null
        super.onDestroy()
    }

    // ---------------------------------------------------------------- Aktionen

    private fun onStartClicked() {
        val html = etHtml.text.toString().trim()
        if (html.isEmpty()) {
            toast("Bitte HTML-Code einfügen – oder „Beispiel einfügen“ nutzen")
            return
        }
        if (!html.contains("<") || !html.contains(">")) {
            toast("Das sieht nicht nach HTML aus – bitte vollständiges HTML einfügen")
            return
        }
        if (!EngineHost.isModelLoaded) {
            androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("Kein Modell geladen")
                .setMessage("Die HTML-App kann die KI-API erst nutzen, wenn ein Modell geladen ist (Button „Modelle“).\n\nTrotzdem starten?")
                .setPositiveButton("Trotzdem starten") { _, _ -> launchRunner(html) }
                .setNegativeButton("Abbrechen", null)
                .show()
            return
        }
        launchRunner(html)
    }

    private fun launchRunner(html: String) {
        val server = ServerHost.server
        if (server == null) {
            ServerHost.start(applicationContext)
            if (ServerHost.server == null) {
                toast("Server konnte nicht gestartet werden")
                return
            }
        }
        ServerHost.server?.currentHtml = html
        try { lastHtmlFile().writeText(html) } catch (_: Throwable) {}
        startActivity(Intent(this, HtmlRunnerActivity::class.java))
    }

    @SuppressLint("SetTextI18n")
    private fun showSysPrompt() {
        val pad = (16 * resources.displayMetrics.density).toInt()
        val scroll = ScrollView(this)
        val tv = TextView(this).apply {
            textSize = 11f
            setTextColor(Color.WHITE)
            setText(SysPrompt.build())
            setPadding(pad, pad / 2, pad, pad / 2)
            setTextIsSelectable(true)
            typeface = android.graphics.Typeface.MONOSPACE
        }
        scroll.addView(tv)
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Systemprompt für externe KI (ChatGPT, Claude, …)")
            .setMessage("Diesen Text an den Anfang des Chats mit der externen KI setzen. Sie weiß dann, wie sie die HTML-App schreibt und an die lokale API anbindet.")
            .setView(scroll)
            .setPositiveButton("Kopieren") { _, _ ->
                val cm = getSystemService(ClipboardManager::class.java)
                cm?.setPrimaryClip(ClipData.newPlainText("lfm3-systemprompt", SysPrompt.build()))
                toast("Systemprompt kopiert – in die externe KI einfügen")
            }
            .setNegativeButton("Schließen", null)
            .show()
    }

    // ---------------------------------------------------------------- Modell-Liste

    @SuppressLint("SetTextI18n", "InflateParams")
    private fun buildModelRows() {
        val inf = layoutInflater
        for (m in ModelCatalog.MODELS) {
            val v = inf.inflate(R.layout.item_launcher_model, null)
            val rv = RowViews(
                root = v,
                name = v.findViewById(R.id.m_name),
                sub = v.findViewById(R.id.m_sub),
                state = v.findViewById(R.id.m_state),
                progress = v.findViewById(R.id.m_prog)
            )
            rv.name.text = m.displayName
            v.setOnClickListener {
                val msg = ModelTasks.onRowTap(applicationContext, m)
                if (msg != null) toast(msg)
                refreshAll()
            }
            v.setOnLongClickListener {
                val p = ModelTasks.cyclePrecision(applicationContext, m)
                toast("Präzision von ${m.displayName}: $p")
                refreshAll()
                true
            }
            modelList.addView(v)
            rows[m.id] = rv
        }
    }

    @SuppressLint("SetTextI18n")
    private fun refreshAll() {
        // Kopfzeile
        val port = ServerHost.port
        val model = EngineHost.currentModel
        tvServer.text = when {
            !ServerHost.isRunning -> "Server: FEHLER"
            model != null -> "Server: 127.0.0.1:$port · Modell: ${model.displayName} [${EngineHost.currentPrecision}]"
            ModelTasks.loadingModelId != null -> "Server: 127.0.0.1:$port · Modell wird geladen…"
            else -> "Server: 127.0.0.1:$port · kein Modell geladen"
        }
        statusDot.setBackgroundColor(
            when {
                EngineHost.isModelLoaded -> Color.parseColor("#37E5C8")
                ModelTasks.loadingModelId != null -> Color.parseColor("#FFC66B")
                else -> Color.parseColor("#9AA3B8")
            })

        // Modell-Zeilen
        for (m in ModelCatalog.MODELS) {
            val rv = rows[m.id] ?: continue
            val prec = ModelTasks.precisionInfo(applicationContext, m)
            val state = ModelTasks.stateOf(applicationContext, m)
            val ds = ModelTasks.downloadState(m, applicationContext)

            rv.sub.text = "${m.kind.label} · ${prec.id} · ${ModelCatalog.humanBytes(m.totalBytes(prec.id))}"
            when (state) {
                ModelTasks.State.NOT_INSTALLED -> {
                    rv.state.text = "⬇ Download"
                    rv.state.setTextColor(Color.parseColor("#9AA3B8"))
                    rv.progress.visibility = View.GONE
                }
                ModelTasks.State.DOWNLOADING -> {
                    rv.state.text = "⏳ ${ds?.pct ?: 0} %"
                    rv.state.setTextColor(Color.parseColor("#FFC66B"))
                    rv.progress.visibility = View.VISIBLE
                    rv.progress.progress = ds?.pct ?: 0
                }
                ModelTasks.State.INSTALLED -> {
                    rv.state.text = "▶ Laden"
                    rv.state.setTextColor(Color.parseColor("#7C5CFF"))
                    rv.progress.visibility = View.GONE
                }
                ModelTasks.State.LOADING -> {
                    rv.state.text = "… lädt"
                    rv.state.setTextColor(Color.parseColor("#FFC66B"))
                    rv.progress.visibility = View.VISIBLE
                    rv.progress.isIndeterminate = true
                }
                ModelTasks.State.LOADED -> {
                    rv.state.text = "● AKTIV"
                    rv.state.setTextColor(Color.parseColor("#37E5C8"))
                    rv.progress.visibility = View.GONE
                }
            }
        }

        ModelTasks.lastMessage?.let { tvServer.contentDescription = it }
    }

    // ---------------------------------------------------------------- Helpers

    private fun updateHtmlInfo() {
        val n = etHtml.text?.length ?: 0
        tvHtmlInfo.text = if (n == 0) "leer"
        else Formatter.formatShortFileSize(this, etHtml.text.toString().toByteArray(Charsets.UTF_8).size.toLong())
    }

    private fun lastHtmlFile(): File =
        File(getExternalFilesDir(null) ?: filesDir, "last_app.html")

    private fun toast(msg: String) =
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    /** Kleines Banner mit dem letzten Absturz-Text, per Klick entfernbar. */
    @SuppressLint("SetTextI18n")
    private fun showCrashBanner(crashText: String) {
        try {
            val root = findViewById<ViewGroup>(R.id.root)
            val banner = TextView(this).apply {
                setTextColor(Color.parseColor("#FFD9D9"))
                setBackgroundColor(Color.parseColor("#AA5A1F1F"))
                textSize = 11f
                setPadding(24, 16, 24, 16)
                val short = crashText.lineSequence().take(3).joinToString("\n")
                setText("Letzter Absturz erkannt:\n$short …\n(Tippen zum Ansehen/Kopieren)")
                setOnClickListener { showCrashDialog(crashText) }
            }
            root.addView(banner, 0, ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        } catch (_: Throwable) {
        }
    }

    private fun showCrashDialog(text: String) {
        try {
            val pad = (16 * resources.displayMetrics.density).toInt()
            val scroll = ScrollView(this)
            val tv = TextView(this).apply {
                textSize = 11f
                setTextColor(Color.WHITE)
                setText(text)
                setPadding(pad, pad / 2, pad, pad / 2)
                setTextIsSelectable(true)
            }
            scroll.addView(tv)
            androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("Letzter Absturz")
                .setView(scroll)
                .setPositiveButton("Kopieren") { _, _ ->
                    val cm = getSystemService(ClipboardManager::class.java)
                    cm?.setPrimaryClip(ClipData.newPlainText("crash", text))
                    Toast.makeText(this, "In Zwischenablage kopiert", Toast.LENGTH_SHORT).show()
                    App.clearCrashLog(this)
                }
                .setNegativeButton("OK") { _, _ -> App.clearCrashLog(this) }
                .show()
        } catch (_: Throwable) {
        }
    }

    /** Not-Ausgabe: zeigt jeden Start-Fehler direkt auf dem Bildschirm. */
    @SuppressLint("SetTextI18n")
    private fun showCrashScreen(t: Throwable) {
        try {
            val sw = java.io.StringWriter()
            t.printStackTrace(java.io.PrintWriter(sw))
            val stack = sw.toString()
            try {
                java.io.File(App.crashLogDir(this), "crash-latest.txt")
                    .writeText("Startup-Fehler:\n$stack")
            } catch (_: Throwable) {}

            val pad = (16 * resources.displayMetrics.density).toInt()
            val scroll = ScrollView(this).apply { setBackgroundColor(Color.parseColor("#0B0F19")) }
            val box = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(pad, pad, pad, pad)
            }
            box.addView(TextView(this).apply {
                text = "App-Fehler (Start fehlgeschlagen)"
                setTextColor(Color.WHITE)
                textSize = 18f
            })
            box.addView(TextView(this).apply {
                setText(stack)
                setTextColor(Color.parseColor("#FF6B6B"))
                textSize = 12f
                setPadding(0, pad, 0, pad)
                setTextIsSelectable(true)
            })
            box.addView(Button(this).apply {
                text = "Fehler kopieren"
                setOnClickListener {
                    val cm = getSystemService(ClipboardManager::class.java)
                    cm?.setPrimaryClip(ClipData.newPlainText("crash", stack))
                    Toast.makeText(this@MainActivity, "Kopiert", Toast.LENGTH_SHORT).show()
                }
            })
            scroll.addView(box)
            setContentView(scroll)
        } catch (_: Throwable) {
            val tv = TextView(this)
            tv.text = "Fehler: ${t.message}"
            setContentView(tv)
        }
    }
}
