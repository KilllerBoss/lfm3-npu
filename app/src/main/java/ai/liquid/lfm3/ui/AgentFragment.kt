package ai.liquid.lfm3.ui

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import ai.liquid.lfm3.R
import ai.liquid.lfm3.engine.EngineHost
import java.util.Locale

/**
 * Agent-Fragment: Bild + Fragen -> strukturierte JSON-Antworten (d1 / VL)
 * bzw. Tool-Calling (Rechner/Uhrzeit) mit Text-Modellen.
 */
class AgentFragment : Fragment(R.layout.fragment_agent) {

    private lateinit var etQuestions: EditText
    private lateinit var tvState: TextView
    private lateinit var imgPreview: ImageView
    private lateinit var adapter: LogAdapter

    private var image: Bitmap? = null
    @Volatile private var running = false

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        EngineHost.init(requireContext())

        etQuestions = view.findViewById(R.id.et_questions)
        tvState = view.findViewById(R.id.tv_agent_state)
        imgPreview = view.findViewById(R.id.img_preview)
        adapter = LogAdapter(results)

        val rv = view.findViewById<RecyclerView>(R.id.rv_results)
        rv.layoutManager = LinearLayoutManager(requireContext())
        rv.adapter = adapter

        view.findViewById<ImageButton>(R.id.btn_pick).setOnClickListener { pick() }
        view.findViewById<View>(R.id.btn_run).setOnClickListener { run() }
    }

    private fun pick() {
        try {
            val i = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "image/*"
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivityForResult(i, 61)
        } catch (t: Throwable) {
            Toast.makeText(requireContext(), "Kein Bild-Picker verfügbar", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 61 && resultCode == Activity.RESULT_OK) {
            val uri = data?.data ?: return
            try {
                requireContext().contentResolver.takePersistableUriPermission(
                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: Throwable) {}
            Thread {
                try {
                    val opts = BitmapFactory.Options().apply { inSampleSize = 2 }
                    val bmp = requireContext().contentResolver.openInputStream(uri)?.use {
                        BitmapFactory.decodeStream(it, null, opts)
                    }
                    image = bmp
                    requireActivity().runOnUiThread {
                        imgPreview.visibility = if (bmp != null) View.VISIBLE else View.GONE
                        imgPreview.setImageBitmap(bmp)
                    }
                } catch (t: Throwable) {
                    requireActivity().runOnUiThread {
                        Toast.makeText(requireContext(), "Bild konnte nicht geladen werden", Toast.LENGTH_SHORT).show()
                    }
                }
            }.start()
        }
    }

    private fun run() {
        if (running) {
            tvState.text = "Läuft bereits…"
            return
        }
        if (!EngineHost.isModelLoaded) {
            tvState.text = "Kein Modell geladen – bitte unter „Modelle“ laden"
            return
        }
        val questions = etQuestions.text.toString().lines()
            .map { it.trim() }.filter { it.isNotEmpty() }
        if (questions.isEmpty()) {
            tvState.text = "Keine Fragen eingegeben"
            return
        }
        running = true
        tvState.text = "0/${questions.size} …"
        val bmp = image

        Thread {
            var idx = 0
            try {
                for (q in questions) {
                    idx++
                    requireActivity().runOnUiThread { tvState.text = "$idx/${questions.size}: $q" }
                    val isJsonTask = EngineHost.currentModel?.kind == ai.liquid.lfm3.engine.ModelKind.DECISION
                    val prompt = if (isJsonTask) {
                        "$q\n\nAntworte ausschließlich als JSON-Objekt mit den Feldern " +
                                "\"answer\", \"confidence\" (0.0-1.0) und \"reason\"."
                    } else {
                        q
                    }
                    var out = EngineHost.runOnce(prompt, bmp, maxTokens = 700)

                    // Tool-Loop (max. 1 Fortsetzung)
                    val toolResult = EngineHost.tryTool(out)
                    if (toolResult != null) {
                        val follow = EngineHost.runOnce(
                            prompt + "\n\n" + toolResult +
                                    "\n\nNutze das Tool-Ergebnis und beantworte jetzt die Frage.",
                            bmp, maxTokens = 400
                        )
                        out = follow
                    }

                    val parsed = extractJson(out)
                    val line = if (parsed != null) {
                        "Frage: $q\n${
                            parsed.replace(Regex("\\s+"), " ").take(500)
                        }"
                    } else {
                        "Frage: $q\n${out.replace(Regex("\\s+"), " ").take(500)}"
                    }
                    requireActivity().runOnUiThread { adapter.append(line) }
                }
                requireActivity().runOnUiThread {
                    tvState.text = "Fertig ($idx Antworten)"
                    running = false
                }
            } catch (t: Throwable) {
                requireActivity().runOnUiThread {
                    tvState.text = "Fehler: ${t.message}"
                    adapter.append("Fehler: ${t.message}")
                    running = false
                }
            }
        }.start()
    }

    /** Extrahiert das erste balancierte {...}-JSON aus dem Text. */
    private fun extractJson(text: String): String? {
        val start = text.indexOf('{')
        if (start < 0) return null
        var depth = 0
        var inStr = false
        var esc = false
        for (i in start until text.length) {
            val c = text[i]
            if (esc) { esc = false; continue }
            when {
                c == '\\' && inStr -> esc = true
                c == '"' -> inStr = !inStr
                !inStr && c == '{' -> depth++
                !inStr && c == '}' -> {
                    depth--
                    if (depth == 0) {
                        val cand = text.substring(start, i + 1)
                        return try {
                            org.json.JSONObject(cand).toString()
                        } catch (_: Throwable) { cand }
                    }
                }
            }
        }
        return null
    }

    companion object {
        private val results = ArrayList<String>()
    }
}
