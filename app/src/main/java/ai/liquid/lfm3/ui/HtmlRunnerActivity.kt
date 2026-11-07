package ai.liquid.lfm3.ui

import android.Manifest
import android.app.Activity
import android.app.DownloadManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Base64
import android.util.Log
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.ConsoleMessage
import android.webkit.GeolocationPermissions
import android.webkit.JavascriptInterface
import android.webkit.PermissionRequest
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.NotificationCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import ai.liquid.lfm3.BuildConfig
import ai.liquid.lfm3.MainActivity
import ai.liquid.lfm3.R
import ai.liquid.lfm3.server.ServerHost
import org.json.JSONObject

/**
 * Führt die HTML-App aus dem Launcher im Vollbild aus (immersive WebView).
 * Die Seite wird vom lokalen Server geladen (GET /app.html) und kann dort
 * die on-device KI-API nutzen.
 *
 * Geräte-Berechtigungen werden an die HTML-App weitergereicht:
 *  - Kamera & Mikrofon  getUserMedia -> onPermissionRequest (auto-grant)
 *  - Standort           navigator.geolocation -> onGeolocationPermissionsShowPrompt
 *  - Datei-Uploads      <input type="file"> -> onShowFileChooser
 *  - Downloads          -> System-Download-Manager / MediaStore
 *  - Extras (Benachrichtigung, Vibration, Teilen, Bildschirm wachhalten,
 *    Berechtigungsabfragen) über die JS-Bridge window.Lfm
 *
 * Zurück-Geste beendet die Vollbild-Ansicht.
 */
class HtmlRunnerActivity : AppCompatActivity() {

    private lateinit var webView: WebView

    // ------------------------------------------------------- Berechtigungen

    /** Offene getUserMedia-Anfrage des WebView. */
    private var pendingWebPerm: PermissionRequest? = null
    /** Offene Standort-Anfrage (Origin + Callback). */
    private var pendingGeo: Pair<String, GeolocationPermissions.Callback>? = null
    /** Offene Anfrage aus der JS-Bridge (Bridge-Namen, z.B. ["camera","location"]). */
    private var pendingBridge: List<String>? = null

    private val permLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        val web = pendingWebPerm; pendingWebPerm = null
        val geo = pendingGeo; pendingGeo = null
        val bridge = pendingBridge; pendingBridge = null
        when {
            web != null -> {
                val grantable = web.resources.filter { r ->
                    grants.any { (perm, ok) -> ok && permToResource(perm) == r }
                }
                if (grantable.isNotEmpty()) web.grant(grantable.toTypedArray()) else web.deny()
            }
            geo != null -> {
                val (origin, cb) = geo
                cb.invoke(origin, grants.values.any { it }, false)
            }
            bridge != null -> {
                val detail = JSONObject()
                for (name in bridge) {
                    val perms = Permissions.forBridgeName(name)
                    detail.put(name, if (perms.isEmpty()) true
                               else perms.any { grants[it] == true })
                }
                dispatchPermEvent(detail)
            }
        }
    }

    private fun dispatchPermEvent(detail: JSONObject) {
        runOnUiThread {
            try {
                webView.evaluateJavascript(
                    "window.dispatchEvent(new CustomEvent('lfm3-permissions',{detail:${detail}}))", null)
            } catch (_: Throwable) {}
        }
    }

    // ------------------------------------------------------- Datei-Upload

    private var fileCallback: ValueCallback<Array<Uri>>? = null
    private val fileChooser = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()) { res ->
        val cb = fileCallback
        fileCallback = null
        cb?.onReceiveValue(
            if (res.resultCode == Activity.RESULT_OK)
                WebChromeClient.FileChooserParams.parseResult(res.resultCode, res.data)
            else null
        )
    }

    // ------------------------------------------------------- Lifecycle

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            setContentView(R.layout.activity_html_runner)

            // Immersive Vollbild-Ansicht: Systemleisten ausblenden
            WindowCompat.setDecorFitsSystemWindows(window, false)
            val controller = WindowInsetsControllerCompat(window,
                findViewById(android.R.id.content))
            controller.hide(WindowInsetsCompat.Type.systemBars())
            controller.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE

            webView = findViewById(R.id.web_view)
            webView.settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                mediaPlaybackRequiresUserGesture = false
                allowFileAccess = false
                allowContentAccess = false
                cacheMode = WebSettings.LOAD_DEFAULT
                useWideViewPort = true
                loadWithOverviewMode = true
                mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
                // navigator.geolocation in HTML-Apps aktivieren
                setGeolocationEnabled(true)
            }
            webView.setBackgroundColor(android.graphics.Color.BLACK)

            // JS-Bridge für Funktionen, die reines Web nicht kann
            webView.addJavascriptInterface(LfmBridge(), "Lfm")

            webView.webViewClient = object : WebViewClient() {
                override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                    val url = request.url
                    // Alles, was NICHT der lokale Server ist, im Browser öffnen
                    return if (url.host == "127.0.0.1") {
                        false
                    } else {
                        try {
                            startActivity(Intent(Intent.ACTION_VIEW, url))
                        } catch (_: Throwable) {
                        }
                        true
                    }
                }
            }
            webView.webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(m: ConsoleMessage): Boolean {
                    Log.d("LFM3-WebView", "${m.message()} (${m.sourceId()}:${m.lineNumber()})")
                    return true
                }

                override fun onShowFileChooser(
                    view: WebView, cb: ValueCallback<Array<Uri>>,
                    params: FileChooserParams
                ): Boolean {
                    fileCallback?.onReceiveValue(null)
                    fileCallback = cb
                    return try {
                        fileChooser.launch(params.createIntent())
                        true
                    } catch (_: Throwable) {
                        fileCallback = null
                        false
                    }
                }

                // getUserMedia (Kamera/Mikrofon): fehlende Android-Berechtigungen
                // werden direkt nachgefragt, dann an den WebView weitergegeben
                override fun onPermissionRequest(req: PermissionRequest) {
                    runOnUiThread {
                        val needed = req.resources.mapNotNull { resourceToPermission(it) }
                        if (needed.isEmpty()) { req.deny(); return@runOnUiThread }
                        val missing = needed.filter {
                            checkSelfPermission(it) != android.content.pm.PackageManager.PERMISSION_GRANTED
                        }
                        if (missing.isEmpty()) {
                            req.grant(req.resources)
                        } else {
                            pendingWebPerm = req
                            permLauncher.launch(missing.toTypedArray())
                        }
                    }
                }

                // navigator.geolocation: Berechtigung an HTML-Apps weiterreichen
                override fun onGeolocationPermissionsShowPrompt(
                    origin: String, callback: GeolocationPermissions.Callback
                ) {
                    val fine = Permissions.granted(this@HtmlRunnerActivity,
                        Manifest.permission.ACCESS_FINE_LOCATION)
                    val coarse = Permissions.granted(this@HtmlRunnerActivity,
                        Manifest.permission.ACCESS_COARSE_LOCATION)
                    if (fine || coarse) callback.invoke(origin, true, false)
                    else {
                        pendingGeo = origin to callback
                        permLauncher.launch(arrayOf(
                            Manifest.permission.ACCESS_FINE_LOCATION,
                            Manifest.permission.ACCESS_COARSE_LOCATION))
                    }
                }
            }

            // Downloads aus HTML-Apps (href-Download, Blob-URLs als data:-Fallback)
            webView.setDownloadListener { url, _, contentDisposition, mimeType, _ ->
                handleDownload(url, contentDisposition, mimeType)
            }

            val ts = System.currentTimeMillis()
            webView.loadUrl("${ServerHost.baseUrl}/app.html?ts=$ts")

            onBackPressedDispatcher.addCallback(this, object : androidx.activity.OnBackPressedCallback(true) {
                override fun handleOnBackPressed() {
                    if (webView.canGoBack()) webView.goBack() else finish()
                }
            })
        } catch (t: Throwable) {
            // Auch hier nicht abstürzen: Fehler als Text anzeigen
            val tv = android.widget.TextView(this)
            tv.text = "Runner-Fehler: ${t.message}"
            tv.setPadding(48, 48, 48, 48)
            tv.setTextColor(android.graphics.Color.WHITE)
            setContentView(tv)
        }
    }

    // ================================================================ Downloads

    private fun handleDownload(url: String, contentDisposition: String?, mimeType: String?) {
        try {
            when {
                url.startsWith("http://") || url.startsWith("https://") -> {
                    val name = downloadName(url, contentDisposition, mimeType)
                    val req = DownloadManager.Request(Uri.parse(url))
                        .setMimeType(mimeType)
                        .setTitle(name)
                        .setDescription("LFM3 HTML-App Download")
                        .setNotificationVisibility(
                            DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
                        .setDestinationInExternalPublicDir(Environment.DIRECTORY_DOWNLOADS, name)
                    (getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager).enqueue(req)
                    toast("Download gestartet: $name")
                }
                url.startsWith("data:") -> {
                    saveDataUri(url, mimeType)
                    toast("Datei im Download-Ordner gespeichert")
                }
                else -> toast("Download-Typ nicht unterstützt")
            }
        } catch (t: Throwable) {
            toast("Download fehlgeschlagen: ${t.message ?: t.javaClass.simpleName}")
        }
    }

    private fun downloadName(url: String, contentDisposition: String?, mimeType: String?): String {
        contentDisposition?.let {
            val m = Regex("filename\\s*=\\s*\"?([^\";]+)\"?", RegexOption.IGNORE_CASE).find(it)
            if (m != null) return m.groupValues[1].trim().ifBlank { "lfm3-download" }
        }
        val last = Uri.parse(url).lastPathSegment
        if (!last.isNullOrBlank() && last.contains('.')) return last
        val ext = extensionFor(mimeType)
        return "lfm3-${System.currentTimeMillis()}$ext"
    }

    /** data:[mime][;base64],payload -> MediaStore Downloads (Download-Ordner). */
    private fun saveDataUri(url: String, fallbackMime: String?) {
        val comma = url.indexOf(',')
        if (comma < 0) throw IllegalArgumentException("Ungültige Data-URL")
        val header = url.substring(5, comma)
        val mime = header.substringBefore(';').ifBlank {
            fallbackMime?.ifBlank { null } ?: "application/octet-stream"
        }
        val data = if (header.contains("base64", ignoreCase = true))
            Base64.decode(url.substring(comma + 1), Base64.DEFAULT)
        else url.substring(comma + 1).toByteArray(Charsets.UTF_8)

        val name = "lfm3-${System.currentTimeMillis()}${extensionFor(mime)}"
        val cv = ContentValues().apply {
            put(android.provider.MediaStore.Downloads.DISPLAY_NAME, name)
            put(android.provider.MediaStore.Downloads.MIME_TYPE, mime)
            put(android.provider.MediaStore.Downloads.IS_PENDING, 1)
        }
        val resolver = contentResolver
        val uri = resolver.insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv)
            ?: throw IllegalStateException("Speicher nicht verfügbar")
        resolver.openOutputStream(uri)?.use { it.write(data) }
        cv.clear()
        cv.put(android.provider.MediaStore.Downloads.IS_PENDING, 0)
        resolver.update(uri, cv, null, null)
    }

    private fun extensionFor(mime: String?): String = when {
        mime == null -> ".bin"
        mime.contains("png") -> ".png"
        mime.contains("jpeg") || mime.contains("jpg") -> ".jpg"
        mime.contains("webp") -> ".webp"
        mime.contains("gif") -> ".gif"
        mime.contains("svg") -> ".svg"
        mime.contains("json") -> ".json"
        mime.contains("html") -> ".html"
        mime.contains("csv") -> ".csv"
        mime.contains("pdf") -> ".pdf"
        mime.contains("zip") -> ".zip"
        mime.contains("audio/mpeg") || mime.contains("mp3") -> ".mp3"
        mime.contains("wav") -> ".wav"
        mime.contains("ogg") -> ".ogg"
        mime.contains("webm") -> ".webm"
        mime.contains("mp4") -> ".mp4"
        mime.startsWith("text/") -> ".txt"
        else -> ".bin"
    }

    // ================================================================ JS-Bridge

    private fun toast(msg: String) =
        android.widget.Toast.makeText(this, msg, android.widget.Toast.LENGTH_SHORT).show()

    /**
     * Native Extras für HTML-Apps, die reines Web nicht kann.
     * In JS: window.Lfm.notify("Titel", "Text") usw.
     */
    inner class LfmBridge {

        /** System-Benachrichtigung zeigen. Return: "ok" | "no-permission" | "error". */
        @JavascriptInterface
        fun notify(title: String, body: String): String {
            if (!Permissions.notificationsGranted(this@HtmlRunnerActivity)) return "no-permission"
            return try {
                val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                if (Build.VERSION.SDK_INT >= 26) {
                    nm.createNotificationChannel(NotificationChannel(
                        CHANNEL_ID, "LFM3 HTML-Apps", NotificationManager.IMPORTANCE_DEFAULT))
                }
                val pi = PendingIntent.getActivity(
                    this@HtmlRunnerActivity, 0,
                    Intent(this@HtmlRunnerActivity, MainActivity::class.java),
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
                val n = NotificationCompat.Builder(this@HtmlRunnerActivity, CHANNEL_ID)
                    .setSmallIcon(android.R.drawable.ic_dialog_info)
                    .setContentTitle(title.ifBlank { "LFM3 App" })
                    .setContentText(body)
                    .setStyle(NotificationCompat.BigTextStyle().bigText(body))
                    .setAutoCancel(true)
                    .setContentIntent(pi)
                    .build()
                nm.notify((System.currentTimeMillis() and 0x7FFFFFFF).toInt(), n)
                "ok"
            } catch (_: Throwable) { "error" }
        }

        /** Gerät vibrieren lassen (Millisekunden, 10–10000). Return: "ok" | "error". */
        @JavascriptInterface
        fun vibrate(ms: Long): String = try {
            val v: Vibrator =
                if (Build.VERSION.SDK_INT >= 31)
                    (getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
                else
                    @Suppress("DEPRECATION") (getSystemService(Context.VIBRATOR_SERVICE) as Vibrator)
            if (!v.hasVibrator()) "error"
            else {
                v.vibrate(VibrationEffect.createOneShot(
                    ms.coerceIn(10, 10_000), VibrationEffect.DEFAULT_AMPLITUDE))
                "ok"
            }
        } catch (_: Throwable) { "error" }

        /** true/false, ob die HTML-App die Berechtigung hat (camera, microphone, location, notifications, media, bluetooth). */
        @JavascriptInterface
        fun hasPermission(name: String): Boolean {
            val perms = Permissions.forBridgeName(name)
            if (perms.isEmpty()) return true // Normal-Berechtigungen (Internet, Vibration, …)
            return perms.any { Permissions.granted(this@HtmlRunnerActivity, it) }
        }

        /**
         * Berechtigungen anfordern (z.B. Lfm.requestPermissions("camera,location")).
         * Ergebnis kommt als Event: window.addEventListener("lfm3-permissions", e => e.detail.camera).
         * Return: "granted" (schon alles da) | "requested" | "no-op".
         */
        @JavascriptInterface
        fun requestPermissions(names: String): String {
            val list = names.split(',').map { it.trim() }
                .filter { it.isNotEmpty() }.distinct()
            if (list.isEmpty()) return "no-op"
            val missing = list.flatMap { Permissions.forBridgeName(it) }.distinct()
                .filter { !Permissions.granted(this@HtmlRunnerActivity, it) }
            return if (missing.isEmpty()) {
                val detail = JSONObject()
                for (name in list) detail.put(name, true)
                dispatchPermEvent(detail)
                "granted"
            } else {
                runOnUiThread {
                    pendingBridge = list
                    permLauncher.launch(missing.toTypedArray())
                }
                "requested"
            }
        }

        /** Bildschirm wachhalten, solange die App läuft. */
        @JavascriptInterface
        fun keepScreenOn(on: Boolean) {
            runOnUiThread {
                if (on) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }

        /** Android-Teilen-Dialog (z.B. Ergebnis als Text teilen). */
        @JavascriptInterface
        fun share(text: String, title: String) {
            runOnUiThread {
                try {
                    val i = Intent(Intent.ACTION_SEND).apply {
                        type = "text/plain"
                        putExtra(Intent.EXTRA_TEXT, text)
                        putExtra(Intent.EXTRA_TITLE, title)
                    }
                    startActivity(Intent.createChooser(i, title.ifBlank { "Teilen" }))
                } catch (_: Throwable) {}
            }
        }

        /** Vollbild-App schließen (zurück zum Launcher). */
        @JavascriptInterface
        fun exitApp() { runOnUiThread { finish() } }

        /** Version der Host-App, z.B. "1.3.0". */
        @JavascriptInterface
        fun appVersion(): String = BuildConfig.VERSION_NAME
    }

    // ================================================================ WebView-Mapping

    private fun permToResource(res: String): String? = when (res) {
        PermissionRequest.RESOURCE_AUDIO_CAPTURE -> "AUDIO_CAPTURE"
        PermissionRequest.RESOURCE_VIDEO_CAPTURE -> "VIDEO_CAPTURE"
        else -> null
    }

    private fun resourceToPermission(res: String): String? = when (res) {
        PermissionRequest.RESOURCE_AUDIO_CAPTURE -> Manifest.permission.RECORD_AUDIO
        PermissionRequest.RESOURCE_VIDEO_CAPTURE -> Manifest.permission.CAMERA
        else -> null
    }

    override fun onPause() {
        super.onPause()
        try { webView.onPause() } catch (_: Throwable) {}
    }

    override fun onResume() {
        super.onResume()
        try { webView.onResume() } catch (_: Throwable) {}
    }

    override fun onDestroy() {
        pendingWebPerm = null
        pendingGeo = null
        pendingBridge = null
        try {
            (webView.parent as? ViewGroup)?.removeView(webView)
            webView.destroy()
        } catch (_: Throwable) {}
        super.onDestroy()
    }

    companion object {
        private const val CHANNEL_ID = "lfm3_html_apps"
    }
}
