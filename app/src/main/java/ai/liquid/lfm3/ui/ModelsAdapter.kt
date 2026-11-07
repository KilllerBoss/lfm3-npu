package ai.liquid.lfm3.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import ai.liquid.lfm3.R
import ai.liquid.lfm3.engine.ModelCatalog
import ai.liquid.lfm3.engine.ModelInfo
import ai.liquid.lfm3.engine.ModelManager

/**
 * Modell-Liste: Präzision wählen, Download mit Fortschritt, Laden, Löschen.
 */
class ModelsAdapter(
    private val items: List<ModelInfo>,
    private val listener: Listener
) : RecyclerView.Adapter<ModelsAdapter.VH>() {

    interface Listener {
        fun onCyclePrecision(model: ModelInfo, current: String): String
        fun onAction(model: ModelInfo, precId: String)
        fun onDelete(model: ModelInfo, precId: String)
        fun stateText(model: ModelInfo, precId: String): String
        fun progress(model: ModelInfo, precId: String): Triple<Int, Long, Long>? // (pct, done, total) oder null
        fun isBusy(model: ModelInfo): Boolean
    }

    class VH(v: View) : RecyclerView.ViewHolder(v) {
        val tvName: TextView = v.findViewById(R.id.tv_name)
        val tvKind: TextView = v.findViewById(R.id.tv_kind)
        val tvDesc: TextView = v.findViewById(R.id.tv_desc)
        val tvPrecisions: TextView = v.findViewById(R.id.tv_precisions)
        val btnPrec: Button = v.findViewById(R.id.btn_prec)
        val btnAction: Button = v.findViewById(R.id.btn_action)
        val tvState: TextView = v.findViewById(R.id.tv_state)
        val pb: ProgressBar = v.findViewById(R.id.pb_dl)
    }

    private val selectedPrecision = HashMap<String, String>()

    fun precisionOf(m: ModelInfo): String =
        selectedPrecision[m.id] ?: ModelCatalog.DEFAULT_PRECISION

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH =
        VH(LayoutInflater.from(parent.context).inflate(R.layout.item_model, parent, false))

    override fun getItemCount(): Int = items.size

    override fun onBindViewHolder(h: VH, pos: Int) {
        val m = items[pos]
        val precId = precisionOf(m)
        h.tvName.text = m.displayName
        h.tvKind.text = m.kind.label
        h.tvDesc.text = m.desc

        val sb = StringBuilder()
        for (p in m.precisions) {
            sb.append(p.id).append(" ").append(ModelCatalog.humanBytes(m.totalBytes(p.id)))
            if (p != m.precisions.last()) sb.append(" · ")
        }
        h.tvPrecisions.text = sb

        h.btnPrec.text = precId
        h.btnPrec.setOnClickListener {
            val next = listener.onCyclePrecision(m, precId)
            selectedPrecision[m.id] = next
            notifyItemChanged(pos)
        }

        h.btnAction.setOnClickListener {
            if (!listener.isBusy(m)) listener.onAction(m, precId)
        }
        h.btnAction.setOnLongClickListener {
            if (!listener.isBusy(m)) { listener.onDelete(m, precId); true } else false
        }

        val prog = listener.progress(m, precId)
        if (prog != null) {
            h.pb.visibility = View.VISIBLE
            h.pb.isIndeterminate = prog.first <= 0
            if (prog.first > 0) h.pb.progress = prog.first
            h.tvState.text = String.format(java.util.Locale.GERMANY,
                "%s · %d%% (%s / %s)",
                "lädt…", prog.first,
                ModelCatalog.humanBytes(prog.second),
                ModelCatalog.humanBytes(prog.third))
        } else {
            h.pb.visibility = View.GONE
            h.tvState.text = listener.stateText(m, precId)
        }

        h.btnAction.text = when {
            listener.isBusy(m) -> "Abbrechen"
            ModelManager.isInstalled(h.itemView.context, m, m.precision(precId)!!) ->
                h.itemView.context.getString(R.string.load)
            else -> h.itemView.context.getString(R.string.download)
        }
    }
}
