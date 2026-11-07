package ai.liquid.lfm3

import android.app.Application
import android.content.Context
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Application-Klasse: installiert einen globalen Crash-Handler, der jeden
 * Absturz auf die Festplatte schreibt, damit er im UI sichtbar gemacht
 * werden kann (statt stillen Beenden beim Start).
 */
class App : Application() {

    companion object {
        @JvmStatic lateinit var instance: App
            private set

        @JvmStatic var lastCrash: String? = null
            private set

        fun crashLogDir(ctx: Context): File =
            File(ctx.getExternalFilesDir(null) ?: ctx.filesDir, "logs").apply { mkdirs() }

        fun clearCrashLog(ctx: Context) {
            lastCrash = null
            try { File(crashLogDir(ctx), "crash-latest.txt").delete() } catch (_: Throwable) {}
        }
    }

    override fun onCreate() {
        instance = this
        super.onCreate()
        lastCrash = readLatestCrash()

        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            try {
                val sw = StringWriter()
                e.printStackTrace(PrintWriter(sw))
                val ts = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())
                val text = "Absturz um $ts in Thread '${t.name}'\n${sw}"
                File(crashLogDir(this), "crash-latest.txt").writeText(text)
                val safe = ts.replace(" ", "_").replace(":", "-")
                File(crashLogDir(this), "crash-$safe.txt").writeText(text)
            } catch (_: Throwable) {
            }
            previous?.uncaughtException(t, e)
        }
    }

    private fun readLatestCrash(): String? = try {
        val f = File(crashLogDir(this), "crash-latest.txt")
        if (f.exists()) f.readText().take(4000) else null
    } catch (_: Throwable) {
        null
    }
}
