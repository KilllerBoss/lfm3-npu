package ai.liquid.lfm3.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import ai.liquid.lfm3.R
import ai.liquid.lfm3.engine.EngineHost
import java.util.Locale

/**
 * Live-Sprachmodus: Spracheingabe (System-Erkennung) -> LLM -> Sprachausgabe (TTS).
 */
class LiveFragment : Fragment(R.layout.fragment_live) {

    private lateinit var tvState: TextView
    private lateinit var btnTalk: Button
    private lateinit var adapter: LogAdapter

    private var recognizer: SpeechRecognizer? = null
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var running = false

    /** UI-Thread-Helper ohne Crash-Risiko bei Detached-State (Tab-Wechsel mitten
     *  im Sprach-Loop): Activity wird gecaptured, Fehler im Block werden
     *  geschluckt statt die App zu töten. */
    private fun ui(block: () -> Unit) {
        val act = activity ?: return
        act.runOnUiThread { try { block() } catch (_: Throwable) {} }
    }

    /** Detach-sicheres getString (liefert "", wenn das Fragment nicht mehr
     *  attached ist - vorher würde getString() hier crashen). */
    private fun s(id: Int): String = if (isAdded) getString(id) else ""

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        EngineHost.init(requireContext())

        tvState = view.findViewById(R.id.tv_state)
        btnTalk = view.findViewById(R.id.btn_talk)

        adapter = LogAdapter(lines)
        val rv = view.findViewById<RecyclerView>(R.id.rv_live)
        rv.layoutManager = LinearLayoutManager(requireContext()).apply { stackFromEnd = true }
        rv.adapter = adapter

        btnTalk.setOnClickListener {
            if (running) stopAll() else startConversation()
        }

        tts = TextToSpeech(requireContext()) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) tts?.language = Locale.GERMANY
        }
    }

    private fun startConversation() {
        if (!EngineHost.isModelLoaded) {
            tvState.text = "Kein Modell geladen – bitte unter „Modelle“ laden"
            return
        }
        running = true
        btnTalk.text = getString(R.string.live_stop)
        listen()
    }

    private fun stopAll() {
        running = false
        try { recognizer?.stopListening() } catch (_: Throwable) {}
        try { recognizer?.destroy() } catch (_: Throwable) {}
        recognizer = null
        try { tts?.stop() } catch (_: Throwable) {}
        if (this::btnTalk.isInitialized) btnTalk.text = s(R.string.live_start)
        if (this::tvState.isInitialized) tvState.text = s(R.string.live_state_idle)
    }

    private fun listen() {
        if (!running || !isAdded) { running = false; return }
        // CRASH-FIX: kein requireContext()/requireActivity() mehr - listen() wird
        // auch aus TTS-/Erkennungs-Callbacks aufgerufen, die nach dem Tab-Wechsel
        // laufen können.
        val ctx = context ?: run { running = false; return }
        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            val act = activity ?: run { running = false; return }
            ActivityCompat.requestPermissions(act,
                arrayOf(Manifest.permission.RECORD_AUDIO), 51)
            return
        }
        if (!SpeechRecognizer.isRecognitionAvailable(ctx)) {
            tvState.text = "Spracherkennung nicht verfügbar"
            stopAll()
            return
        }
        tvState.text = s(R.string.live_state_listening)
        recognizer = SpeechRecognizer.createSpeechRecognizer(ctx).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onResults(results: Bundle) {
                    val txt = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        ?.firstOrNull().orEmpty()
                    if (!running) return
                    if (txt.isEmpty()) {
                        tvState.text = "Nichts verstanden – nochmal tippen"
                        return
                    }
                    adapter.append("Du: $txt")
                    think(txt)
                }
                override fun onError(error: Int) {
                    if (!running) return
                    tvState.text = "Erkennungsfehler ($error) – nochmal tippen"
                }
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {
                    if (running) tvState.text = s(R.string.live_state_thinking)
                }
                override fun onPartialResults(partialResults: Bundle?) {}
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
            val intent = android.content.Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                    RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "de-DE")
            }
            startListening(intent)
        }
    }

    private fun think(userText: String) {
        if (!isAdded) { running = false; return }
        tvState.text = s(R.string.live_state_thinking)
        val act = activity ?: return

        val hist = ArrayList<EngineHost.Msg>()
        hist.add(EngineHost.Msg("user", userText))
        hist.add(EngineHost.Msg("assistant", ""))

        val sb = StringBuilder()
        EngineHost.chat(
            hist.subList(0, 1),
            null,
            onDelta = { delta, _ ->
                synchronized(sb) { sb.append(delta) }
            },
            onStats = {
                ui {
                    val answer = synchronized(sb) { sb.toString() }.trim()
                    if (answer.isEmpty()) answer_fallback()
                    else speak(answer)
                }
            },
            onError = { err ->
                ui {
                    adapter.append("Fehler: $err")
                    tvState.text = "Fehler – nochmal tippen"
                }
            }
        )
    }

    private fun answer_fallback() {
        adapter.append("Bot: (leere Antwort)")
        tvState.text = "Leere Antwort – nochmal tippen"
    }

    private fun speak(text: String) {
        // Tab gewechselt / Fragment weg: Sprach-Loop sauber stoppen
        if (!isAdded) { running = false; return }
        val short = text.replace(Regex("\\s+"), " ").take(600)
        adapter.append("Bot: $short")
        tvState.text = s(R.string.live_state_speaking)
        if (ttsReady) {
            tts?.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                override fun onDone(utteranceId: String?) {
                    ui {
                        if (running) listen()
                    }
                }
                override fun onError(utteranceId: String?) {}
                override fun onStart(utteranceId: String?) {}
            })
            tts?.speak(short, TextToSpeech.QUEUE_FLUSH, null, "lfm3")
        } else {
            // Ohne TTS direkt wieder zuhören
            listen()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 51 && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            listen()
        } else {
            tvState.text = "Mikrofon-Berechtigung fehlt"
            stopAll()
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        running = false
        try { recognizer?.destroy() } catch (_: Throwable) {}
        try { tts?.shutdown() } catch (_: Throwable) {}
        recognizer = null
        tts = null
    }

    companion object {
        private val lines = ArrayList<String>()
    }
}
