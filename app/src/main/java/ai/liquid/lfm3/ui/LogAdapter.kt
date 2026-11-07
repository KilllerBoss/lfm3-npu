package ai.liquid.lfm3.ui

import android.view.LayoutInflater
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import ai.liquid.lfm3.R

/** Einfache Textzeilen (Logs, Live-Protokoll, Agent-Ergebnisse). */
class LogAdapter(private val items: MutableList<String>) :
    RecyclerView.Adapter<LogAdapter.VH>() {

    class VH(v: android.view.View) : RecyclerView.ViewHolder(v) {
        val tv: TextView = v.findViewById(R.id.tv_log)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH =
        VH(LayoutInflater.from(parent.context).inflate(R.layout.item_log, parent, false))

    override fun getItemCount(): Int = items.size

    override fun onBindViewHolder(h: VH, pos: Int) {
        h.tv.text = items[pos]
    }

    fun append(line: String) {
        items.add(line)
        notifyItemInserted(items.size - 1)
    }

    fun replaceLast(line: String) {
        if (items.isEmpty()) { append(line); return }
        items[items.size - 1] = line
        notifyItemChanged(items.size - 1)
    }

    fun clear() {
        items.clear()
        notifyDataSetChanged()
    }
}
