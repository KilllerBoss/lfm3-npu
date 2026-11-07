package ai.liquid.lfm3.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import ai.liquid.lfm3.R
import ai.liquid.lfm3.engine.EngineHost

/**
 * Chat-Adapter: zwei Zelltypen (Nutzer mit Bubble-Violett, Bot mit Think-Bereich).
 */
class ChatAdapter(private val items: MutableList<EngineHost.Msg>) :
    RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    companion object {
        const val TYPE_USER = 0
        const val TYPE_BOT = 1
    }

    var onImageClick: ((Int) -> Unit)? = null

    class UserVH(v: View) : RecyclerView.ViewHolder(v) {
        val img: ImageView = v.findViewById(R.id.img_msg)
        val msg: TextView = v.findViewById(R.id.tv_msg)
    }

    class BotVH(v: View) : RecyclerView.ViewHolder(v) {
        val think: TextView = v.findViewById(R.id.tv_think)
        val msg: TextView = v.findViewById(R.id.tv_msg)
        val stats: TextView = v.findViewById(R.id.tv_stats)
    }

    override fun getItemViewType(position: Int): Int =
        if (items[position].role == "user") TYPE_USER else TYPE_BOT

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inf = LayoutInflater.from(parent.context)
        return if (viewType == TYPE_USER) {
            UserVH(inf.inflate(R.layout.item_msg_user, parent, false))
        } else {
            BotVH(inf.inflate(R.layout.item_msg_bot, parent, false))
        }
    }

    override fun onBindViewHolder(h: RecyclerView.ViewHolder, pos: Int) {
        val m = items[pos]
        if (h is UserVH) {
            h.msg.text = m.text
            if (m.bitmap != null) {
                h.img.visibility = View.VISIBLE
                h.img.setImageBitmap(m.bitmap)
                h.img.setOnClickListener { onImageClick?.invoke(h.bindingAdapterPosition) }
            } else {
                h.img.visibility = View.GONE
            }
        } else if (h is BotVH) {
            if (m.think.isEmpty()) {
                h.think.visibility = View.GONE
            } else {
                h.think.visibility = View.VISIBLE
                h.think.text = m.think
            }
            h.msg.text = m.text.ifEmpty { "…" }
            if (m.stats.isEmpty()) {
                h.stats.visibility = View.GONE
            } else {
                h.stats.visibility = View.VISIBLE
                h.stats.text = m.stats
            }
        }
    }

    override fun getItemCount(): Int = items.size

    fun notifyItemChangedSafe(pos: Int) {
        if (pos in items.indices) notifyItemChanged(pos)
    }
}
