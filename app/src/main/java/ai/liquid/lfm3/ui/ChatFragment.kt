package ai.liquid.lfm3.ui

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import ai.liquid.lfm3.R
import ai.liquid.lfm3.engine.EngineHost
import java.util.Locale

/**
 * Chat-Fragment: Streaming-Antworten, Bildanhang (VL), Spracheingabe.
 */
class ChatFragment : Fragment(R.layout.fragment_chat) {

    private lateinit var adapter: ChatAdapter
    private lateinit var tvModel: TextView
    private lateinit var tvBackend: TextView
    private lateinit var etInput: EditText
    private lateinit var btnSend: ImageButton
    private lateinit var btnAttach: ImageButton
    private lateinit var btnMic: ImageButton

    private var pendingImage: Bitmap? = null
    private var pendingImageUri: Uri? = null
    private var lastUiUpdate = 0L
    private var pendingText = StringBuilder()
    private var pendingThink = StringBuilder()
    private var uiTickScheduled = false

    private var recognizer: SpeechRecognizer? = null
    private var listening = false

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        EngineHost.init(requireContext())

        tvModel = view.findViewById(R.id.tv_model_name)
        tvBackend = view.findViewById(R.id.tv_backend)
        etInput = view.findViewById(R.id.et_input)
        btnSend = view.findViewById(R.id.btn_send)
        btnAttach = view.findViewById(R.id.btn_attach)
        btnMic = view.findViewById(R.id.btn_mic)

        adapter = ChatAdapter(history)
        val rv = view.findViewById<RecyclerView>(R.id.rv_chat)
        rv.layoutManager = LinearLayoutManager(requireContext()).apply {
            stackFromEnd = true
        }
        rv.adapter = adapter

        view.findViewById<ImageButton>(R.id.btn_new_chat).setOnClickListener {
            if (EngineHost.isGenerating) EngineHost.stopGeneration()
            history.clear()
            adapter.notifyDataSetChanged()
        }

        btnSend.setOnClickListener {
            if (EngineHost.isGenerating) {
                EngineHost.stopGeneration()
                return@setOnClickListener
            }
            val text = etInput.text.toString().trim()
            if (text.isEmpty() && pendingImage == null) return@setOnClickListener
            send(text)
        }

        btnAttach.setOnClickListener { pickImage() }
        btnMic.setOnClickListener { toggleMic() }

        // Header aktualisieren
        refreshHeader()
    }

    override fun onResume() {
        super.onResume()
        refreshHeader()
    }

    private fun refreshHeader() {
        tvModel.text = EngineHost.activeModelLabel
        tvBackend.text = if (EngineHost.isModelLoaded) {
            "Backend: ${EngineHost.backendLabel}"
        } else {
            getString(R.string.no_model_loaded)
        }
        btnAttach.visibility = if (EngineHost.currentModel?.needsVision == true) View.VISIBLE else View.GONE
    }

    // ---------------------------------------------------------------- Senden

    private fun send(text: String) {
        etInput.setText("")

        val userMsg = EngineHost.Msg("user", text, bitmap = pendingImage)
        history.add(userMsg)
        adapter.notifyItemInserted(history.size - 1)

        val botMsg = EngineHost.Msg("assistant", "", think = "", stats = "")
        history.add(botMsg)
        adapter.notifyItemInserted(history.size - 1)
        scrollToBottom()

        val botIndex = history.size - 1
        pendingText = StringBuilder()
        pendingThink = StringBuilder()
        lastUiUpdate = 0

        // Kopie der Historie OHNE den leeren Bot-Platzhalter
        val histForPrompt = history.subList(0, history.size - 1).map { it }
        val image = pendingImage
        pendingImage = null
        pendingImageUri = null

        EngineHost.chat(
            histForPrompt,
            image,
            onDelta = { delta, inThink ->
                synchronized(this) {
                    if (inThink) pendingThink.append(delta) else pendingText.append(delta)
                }
                scheduleUiTick(botIndex)
            },
            onStats = { stats ->
                requireActivity().runOnUiThread {
                    val s = String.format(
                        Locale.GERMANY,
                        "%d Tokens · %.1f Tok/s (Prefill %d ms)",
                        stats.tokens, stats.tokensPerSec, stats.prefillMs
                    )
                    if (botIndex < history.size) {
                        history[botIndex].stats = s
                        adapter.notifyItemChangedSafe(botIndex)
                    }
                }
            },
            onError = { err ->
                requireActivity().runOnUiThread {
                    if (botIndex < history.size) {
                        history[botIndex].text =
                            getString(R.string.error, err)
                        adapter.notifyItemChangedSafe(botIndex)
                    }
                    refreshHeader()
                }
            }
        )
    }

    /** UI-Throttle: max. ~30 Updates/s, gebündelt. */
    private fun scheduleUiTick(botIndex: Int) {
        val act = activity ?: return
        val now = System.currentTimeMillis()
        val need = now - lastUiUpdate >= 30
        if (need && !uiTickScheduled) {
            uiTickScheduled = true
            lastUiUpdate = now
            act.runOnUiThread {
                uiTickScheduled = false
                synchronized(this) {
                    if (botIndex < history.size) {
                        history[botIndex].text = pendingText.toString()
                        history[botIndex].think = pendingThink.toString()
                    }
                }
                adapter.notifyItemChangedSafe(botIndex)
                scrollToBottom()
            }
        }
    }

    private fun scrollToBottom() {
        val rv = view?.findViewById<RecyclerView>(R.id.rv_chat) ?: return
        rv.post { rv.scrollToPosition((adapter.itemCount - 1).coerceAtLeast(0)) }
    }

    // ---------------------------------------------------------------- Bild

    private fun pickImage() {
        try {
            val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                addCategory(Intent.CATEGORY_OPENABLE)
                type = "image/*"
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivityForResult(intent, REQ_PICK)
        } catch (t: Throwable) {
            Toast.makeText(requireContext(), "Kein Bild-Picker verfügbar", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_PICK && resultCode == Activity.RESULT_OK) {
            val uri = data?.data ?: return
            try {
                requireContext().contentResolver.takePersistableUriPermission(
                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: Throwable) {}
            pendingImageUri = uri
            Thread {
                try {
                    val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    requireContext().contentResolver.openInputStream(uri)?.use {
                        BitmapFactory.decodeStream(it, null, opts)
                    }
                    var sample = 1
                    val maxDim = 1280
                    while (opts.outWidth / (sample * 2) >= maxDim || opts.outHeight / (sample * 2) >= maxDim) {
                        sample *= 2
                    }
                    val opts2 = BitmapFactory.Options().apply { inSampleSize = sample }
                    val bmp = requireContext().contentResolver.openInputStream(uri)?.use {
                        BitmapFactory.decodeStream(it, null, opts2)
                    }
                    pendingImage = bmp
                    requireActivity().runOnUiThread {
                        Toast.makeText(requireContext(), "Bild angehängt", Toast.LENGTH_SHORT).show()
                    }
                } catch (t: Throwable) {
                    requireActivity().runOnUiThread {
                        Toast.makeText(requireContext(), "Bild konnte nicht geladen werden", Toast.LENGTH_SHORT).show()
                    }
                }
            }.start()
        }
    }

    // ---------------------------------------------------------------- Spracheingabe

    private fun toggleMic() {
        if (listening) {
            recognizer?.stopListening()
            return
        }
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC)
            return
        }
        startListening()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_MIC && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startListening()
        } else if (requestCode == REQ_MIC) {
            Toast.makeText(requireContext(), "Mikrofon-Berechtigung fehlt", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startListening() {
        if (!SpeechRecognizer.isRecognitionAvailable(requireContext())) {
            Toast.makeText(requireContext(), "Spracherkennung nicht verfügbar", Toast.LENGTH_SHORT).show()
            return
        }
        listening = true
        btnMic.setImageResource(R.drawable.ic_stop)
        etInput.hint = getString(R.string.listen_prompt)
        recognizer = SpeechRecognizer.createSpeechRecognizer(requireContext()).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle) {
                    val txt = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull().orEmpty()
                    finishListening()
                    if (txt.isNotEmpty()) {
                        etInput.setText(txt)
                        send(txt)
                    }
                }
                override fun onError(error: Int) {
                    finishListening()
                }
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
            startActivity(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                    RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "de-DE")
            })
            startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                    RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "de-DE")
            })
        }
    }

    private fun finishListening() {
        listening = false
        btnMic.setImageResource(R.drawable.ic_mic)
        etInput.hint = getString(R.string.hint_message)
        try { recognizer?.destroy() } catch (_: Throwable) {}
        recognizer = null
    }

    override fun onDestroyView() {
        super.onDestroyView()
        try { recognizer?.destroy() } catch (_: Throwable) {}
        recognizer = null
    }

    companion object {
        private const val REQ_PICK = 41
        private const val REQ_MIC = 42

        /** Verlauf bleibt über Fragment-Wechsel hinweg bestehen. */
        val history = ArrayList<EngineHost.Msg>()
    }
}
