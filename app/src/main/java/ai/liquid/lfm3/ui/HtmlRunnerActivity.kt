package ai.liquid.lfm3.ui

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.webkit.ConsoleMessage
import android.webkit.PermissionRequest
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import ai.liquid.lfm3.R
import ai.liquid.lfm3.server.ServerHost

/**
 * Führt die HTML-App aus dem Launcher im Vollbild aus (immersive WebView).
 * Die Seite wird vom lokalen Server geladen (GET /app.html) und kann dort
 * die on-device KI-API nutzen. Zurück-Geste beendet die Vollbild-Ansicht.
 */
class HtmlRunnerActivity : AppCompatActivity() {

    private lateinit var webView: WebView

    // Datei-Upload für <input type="file"> (z.B. Bildauswahl für Vision-Modelle)
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

    // getUserMedia (Mikrofon/Kamera) für HTML-Apps, die es brauchen
    private var pendingPerm: PermissionRequest? = null
    private val permLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        val req = pendingPerm
        pendingPerm = null
        if (req != null) {
            val granted = req.resources.filter { r ->
                grants.any { (perm, ok) -> ok && permToResource(perm) == r }
            }
            if (granted.isNotEmpty()) req.grant(granted.toTypedArray()) else req.deny()
        }
    }

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
            }
            webView.setBackgroundColor(android.graphics.Color.BLACK)

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
                            pendingPerm = req
                            permLauncher.launch(missing.toTypedArray())
                        }
                    }
                }
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
        try {
            (webView.parent as? ViewGroup)?.removeView(webView)
            webView.destroy()
        } catch (_: Throwable) {}
        super.onDestroy()
    }
}
