package ai.liquid.lfm3.ui

import android.app.ActivityManager
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import androidx.fragment.app.Fragment
import ai.liquid.lfm3.BuildConfig
import ai.liquid.lfm3.R
import ai.liquid.lfm3.engine.EngineHost

/**
 * Einstellungen & Info: System-Prompt, Thinking, NPU-Fallback,
 * Kontextlänge (1k–16k mit RAM-Hinweis), Temperature, Links.
 */
class InfoFragment : Fragment(R.layout.fragment_info) {

    private lateinit var etSystem: EditText
    private lateinit var swThink: android.widget.Switch
    private lateinit var swFallback: android.widget.Switch
    private lateinit var sbCtx: SeekBar
    private lateinit var sbTemp: SeekBar
    private lateinit var lblCtx: TextView
    private lateinit var lblTemp: TextView
    private lateinit var tvEngine: TextView

    private val ctxValues = intArrayOf(1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        EngineHost.init(requireContext())

        etSystem = view.findViewById(R.id.et_system)
        swThink = view.findViewById(R.id.sw_think)
        swFallback = view.findViewById(R.id.sw_npu_fallback)
        sbCtx = view.findViewById(R.id.sb_ctx)
        sbTemp = view.findViewById(R.id.sb_temp)
        lblCtx = view.findViewById(R.id.lbl_ctx)
        lblTemp = view.findViewById(R.id.lbl_temp)
        tvEngine = view.findViewById(R.id.tv_engine)

        // ---- Werte laden
        etSystem.setText(EngineHost.systemPrompt)
        swThink.isChecked = EngineHost.thinkingMode
        swFallback.isChecked = EngineHost.npuFallback

        val ctxIdx = ctxValues.indexOfFirst { it >= EngineHost.maxContext }
            .let { if (it < 0) 3 else it }
        sbCtx.max = ctxValues.size - 1
        sbCtx.progress = ctxIdx
        updateCtxLabel()

        sbTemp.max = 100
        sbTemp.progress = (EngineHost.temperature * 100).toInt()
        updateTempLabel()

        // ---- Listener
        etSystem.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) EngineHost.systemPrompt = etSystem.text.toString()
        }

        swThink.setOnCheckedChangeListener { _, checked -> EngineHost.thinkingMode = checked }
        swFallback.setOnCheckedChangeListener { _, checked -> EngineHost.npuFallback = checked }

        sbCtx.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                updateCtxLabel()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {
                val v = ctxValues[sbCtx.progress]
                EngineHost.maxContext = v
                if (v >= 8192) {
                    android.widget.Toast.makeText(requireContext(),
                        "Achtung: ≥8k Tokens braucht viel RAM – kann auf Geräten mit wenig " +
                                "Arbeitsspeicher instabil werden", android.widget.Toast.LENGTH_LONG).show()
                }
            }
        })

        sbTemp.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                updateTempLabel()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {
                EngineHost.temperature = sbTemp.progress / 100f
            }
        })

        tvEngine.text = buildEngineInfo()
        tvEngine.textSize = 12f
    }

    private fun updateCtxLabel() {
        val v = ctxValues[sbCtx.progress]
        lblCtx.text = getString(R.string.settings_context, v)
    }

    private fun updateTempLabel() {
        val t = sbTemp.progress / 100f
        val label = getString(R.string.settings_temp, t)
        val extra = if (t <= 0.01f) " (greedy)" else ""
        lblTemp.text = label + extra
    }

    private fun buildEngineInfo(): String {
        val sb = StringBuilder()
        sb.append("App: LFM3 NPU v").append(BuildConfig.VERSION_NAME).append('\n')
        sb.append("Runtime: onnxruntime-android-qnn 1.29.0\n")
        sb.append("Backend: ").append(if (EngineHost.isModelLoaded) EngineHost.backendLabel else "— kein Modell —").append('\n')
        if (EngineHost.currentModel != null) {
            sb.append("Modell: ").append(EngineHost.activeModelLabel).append('\n')
            EngineHost.tokenizer?.let { sb.append(it).append('\n') }
        }
        sb.append("Gerät: ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL)
            .append(" (Android ").append(Build.VERSION.RELEASE).append(")\n")
        sb.append("SoC: ").append(Build.HARDWARE).append('\n')
        try {
            val am = requireContext().getSystemService(ActivityManager::class.java)
            val mi = ActivityManager.MemoryInfo()
            am.getMemoryInfo(mi)
            sb.append("RAM gesamt: ").append(mi.totalMem / (1024 * 1024 * 1024)).append(" GB\n")
            sb.append("RAM frei: ").append(mi.availMem / (1024 * 1024)).append(" MB\n")
        } catch (_: Throwable) {}
        sb.append("Abis: ").append(Build.SUPPORTED_ABIS.joinToString(", ")).append('\n')
        return sb.toString()
    }

    override fun onPause() {
        super.onPause()
        if (this::etSystem.isInitialized) {
            EngineHost.systemPrompt = etSystem.text.toString()
        }
    }
}
