package ai.liquid.lfm3

import android.annotation.SuppressLint
import android.content.ClipData
import android.content.ClipboardManager
import android.graphics.Color
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import ai.liquid.lfm3.ui.AgentFragment
import ai.liquid.lfm3.ui.ChatFragment
import ai.liquid.lfm3.ui.InfoFragment
import ai.liquid.lfm3.ui.LiveFragment
import ai.liquid.lfm3.ui.ModelsFragment
import com.google.android.material.bottomnavigation.BottomNavigationView

class MainActivity : AppCompatActivity() {

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            setContentView(R.layout.activity_main)

            val nav = findViewById<BottomNavigationView>(R.id.bottom_nav)
            if (savedInstanceState == null) {
                switchTo(ChatFragment())
            }
            nav.setOnItemSelectedListener { item ->
                val f: Fragment = when (item.itemId) {
                    R.id.nav_chat -> ChatFragment()
                    R.id.nav_live -> LiveFragment()
                    R.id.nav_agent -> AgentFragment()
                    R.id.nav_models -> ModelsFragment()
                    R.id.nav_info -> InfoFragment()
                    else -> ChatFragment()
                }
                switchTo(f)
                true
            }

            App.lastCrash?.let { showCrashBanner(it) }
        } catch (t: Throwable) {
            showCrashScreen(t)
        }
    }

    private fun switchTo(f: Fragment) {
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, f)
            .commitAllowingStateLoss()
    }

    /** Kleines Banner mit dem letzten Absturz-Text, per Klick entfernbar. */
    @SuppressLint("SetTextI18n")
    private fun showCrashBanner(crashText: String) {
        try {
            val root = findViewById<ViewGroup>(R.id.fragment_container)
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
