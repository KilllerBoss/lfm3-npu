package ai.liquid.lfm3.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat

/**
 * Zentrale Verwaltung aller Laufzeit-Berechtigungen, die HTML-Apps im
 * WebView nutzen können (Kamera, Mikrofon, Standort, Medien, Benach-
 * richtigungen, Bluetooth). Die App fordert sie beim Start an und reicht
 * sie an den WebView weiter – so funktionieren getUserMedia & Co. direkt.
 */
object Permissions {

    /** Alle Laufzeit-Berechtigungen, die HTML-Apps brauchen können. */
    val RUNTIME = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.RECORD_AUDIO,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION,
        Manifest.permission.POST_NOTIFICATIONS,
        Manifest.permission.READ_MEDIA_IMAGES,
        Manifest.permission.READ_MEDIA_VIDEO,
        Manifest.permission.READ_MEDIA_AUDIO,
        Manifest.permission.BLUETOOTH_CONNECT,
        Manifest.permission.BLUETOOTH_SCAN,
    )

    fun granted(ctx: Context, perm: String): Boolean =
        ContextCompat.checkSelfPermission(ctx, perm) == PackageManager.PERMISSION_GRANTED

    fun missing(ctx: Context): List<String> = RUNTIME.filter { !granted(ctx, it) }

    fun grantedCount(ctx: Context): Int = RUNTIME.size - missing(ctx).size

    /** Kurzname für den Nutzer (Deutsch). */
    fun label(perm: String): String = when (perm) {
        Manifest.permission.CAMERA -> "Kamera"
        Manifest.permission.RECORD_AUDIO -> "Mikrofon"
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION -> "Standort"
        Manifest.permission.POST_NOTIFICATIONS -> "Benachrichtigungen"
        Manifest.permission.READ_MEDIA_IMAGES -> "Fotos"
        Manifest.permission.READ_MEDIA_VIDEO -> "Videos"
        Manifest.permission.READ_MEDIA_AUDIO -> "Audio/Musik"
        Manifest.permission.BLUETOOTH_CONNECT -> "Bluetooth (verbinden)"
        Manifest.permission.BLUETOOTH_SCAN -> "Bluetooth (suchen)"
        else -> perm.substringAfterLast('.')
    }

    /** Gruppierte Labels für kompakte Anzeige, z.B. "Standort, Kamera". */
    fun missingLabels(ctx: Context): String =
        missing(ctx).map { label(it) }.distinct().joinToString(", ")

    /** HTML-Bridge-Namen -> Android-Berechtigung(en) für window.Lfm.hasPermission(). */
    fun forBridgeName(name: String): List<String> = when (name.lowercase().trim()) {
        "camera", "video" -> listOf(Manifest.permission.CAMERA)
        "microphone", "mic", "audio" -> listOf(Manifest.permission.RECORD_AUDIO)
        "location", "geolocation" -> listOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION)
        "notifications", "notification" -> listOf(Manifest.permission.POST_NOTIFICATIONS)
        "media", "storage", "photos" -> listOf(
            Manifest.permission.READ_MEDIA_IMAGES,
            Manifest.permission.READ_MEDIA_VIDEO,
            Manifest.permission.READ_MEDIA_AUDIO)
        "bluetooth" -> listOf(
            Manifest.permission.BLUETOOTH_CONNECT,
            Manifest.permission.BLUETOOTH_SCAN)
        "vibrate", "vibration", "internet", "network" -> emptyList() // Normal-Perms, immer erteilt
        else -> emptyList()
    }

    /** Notifications brauchen die Runtime-Prüfung erst ab Android 13. */
    fun notificationsGranted(ctx: Context): Boolean =
        Build.VERSION.SDK_INT < 33 || granted(ctx, Manifest.permission.POST_NOTIFICATIONS)

    /** Android-Berechtigung -> Bridge-Name (für Ergebnis-Events an HTML). */
    fun bridgeName(perm: String): String = when (perm) {
        Manifest.permission.CAMERA -> "camera"
        Manifest.permission.RECORD_AUDIO -> "microphone"
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION -> "location"
        Manifest.permission.POST_NOTIFICATIONS -> "notifications"
        Manifest.permission.READ_MEDIA_IMAGES,
        Manifest.permission.READ_MEDIA_VIDEO,
        Manifest.permission.READ_MEDIA_AUDIO -> "media"
        Manifest.permission.BLUETOOTH_CONNECT,
        Manifest.permission.BLUETOOTH_SCAN -> "bluetooth"
        else -> perm.substringAfterLast('.').lowercase()
    }
}
