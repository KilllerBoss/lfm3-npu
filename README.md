# LFM3 NPU – LiquidAI LFM3 on Hexagon NPU (Android) · HTML-Launcher

Vollständige Android-App, die **alle LiquidAI LFM3 ONNX-Varianten lokal auf dem
Hexagon-NPU (Qualcomm QNN/HTP)** ausführt – als **lokaler Server + HTML-Launcher**:
HTML-Apps, die eine externe KI (ChatGPT, Claude, …) schreibt, laufen im Vollbild-WebView
und nutzen das on-device Modell über eine **OpenAI-kompatible lokale API**
(`POST /v1/chat/completions`, Streaming via SSE, Vision-Input).

> **Hinweis:** Dies ist ein Community-Projekt und steht in keiner Verbindung zu LiquidAI.

## Download

**v1.3.0**: https://github.com/KilllerBoss/lfm3-npu/releases/latest – APK
herunterladen, öffnen, „Unbekannte Apps erlauben“ (mindestens Android 12, arm64).

## Workflow

1. **„Modelle“ aufklappen** → Variante antippen: Download → automatisches Laden
   (langer Tastendruck = Präzision wechseln; geladenes Modell antippen = entladen)
2. **„Systemprompt“** kopieren und einer externen KI geben – sie erfährt daraus,
   wie sie eine einzelne, offlinefähige HTML-App für diese Umgebung schreibt und
   die lokale API korrekt aufruft (Endpunkte, messages-Format, LFM3-Eigenheiten)
3. Generiertes HTML in das Eingabefeld einfügen (oder „Beispiel einfügen“)
4. **„▶ HTML im Vollbild starten“** – Zurück-Geste beendet die Ansicht

## Features

- **Modelle (alle wählbar, Download aus der App):**
  - `LFM3-1.7B-ONNX` – Text-Chat
  - `LFM3-VL-2B-ONNX` – Vision (Bild + Chat)
  - `LFM3-Audio-2B-ONNX` – Audio (Sprach-I/O über System-Speech)
  - `LFM3-Omni-2B-ONNX` – Omni (Text + Bild + Sprache)
  - `d1-2B-ONNX` – strukturierte JSON-Entscheidungen (mit Vision)
- **Alle Quantisierungen:** q4 · q4f32 · q8 · q4f16 · q8f16 · fp16 · fp32
  (je nach Verfügbarkeit im jeweiligen HF-Repo)
- **QNN/HTP-Pure-Backend** (`libQnnHtp.so`, `high_performance`) mit optionalem
  CPU-Fallback; **V68/V69/V73/V75/V79/V81** Hexagon-Architekturen im APK
  (Snapdragon 8 Elite = V79 abgedeckt)
- **Lokaler HTTP-Server (NanoHTTPD, nur 127.0.0.1):**
  - `GET /status`, `GET /models`, `POST /load`, `POST /stop`
  - `POST /v1/chat/completions` – OpenAI-kompatibel (`messages`, `temperature`,
    `max_tokens`, `reasoning`, `stream`); Streaming als SSE
    (`data: {…choices[0].delta.content…}`, Ende `data: [DONE]`)
  - Vision-Input über `image_url`-Parts (data-URL/base64) für VL/Omni/d1;
    Thinking als `reasoning_content`
  - Neue Anfrage bricht die laufende Generierung automatisch ab (Auto-Cancel)
- **HTML-Runner:** immersive Vollbild-WebView (JS, DOM-Storage,
  `<input type=file>` für Bild-Upload, `getUserMedia` nach Freigabe,
  externe Links öffnen den Browser)
- **Alle Geräte-Berechtigungen für HTML-Apps (v1.3.0):** Kamera, Mikrofon,
  Standort, Medien, Benachrichtigungen, Vibration, WAKE_LOCK, Bluetooth –
  Abfrage beim ersten Start, Button „🔒 Berechtigungen“ (X/10), WebView reicht
  `getUserMedia` und `navigator.geolocation` automatisch weiter; Downloads aus
  HTML-Apps landen im Download-Ordner; JS-Bridge **`window.Lfm`**: `notify`,
  `vibrate`, `hasPermission`, `requestPermissions` (Event `lfm3-permissions`),
  `keepScreenOn`, `share`, `exitApp`, `appVersion`
- **Systemprompt-Button:** fertiger, kopierbarer Prompt für externe KIs
  (Umgebung, API, Modellkatalog, Nachrichtenformat, UX-Regeln)
- **Einstellungen:** Präzision pro Modell, CPU-Fallback
- **Absturz-Diagnose:** Startet die App nicht, wird der komplette Stacktrace
  auf dem Bildschirm angezeigt und in `Android/data/ai.liquid.lfm3/files/logs/`
  geschrieben – kein stilles Abstürzen mehr.

## Build

```bash
./gradlew assembleRelease     # oder assembleDebug
# APK: app/build/outputs/apk/release/app-release.apk
```

Anforderungen: JDK 21, Android SDK 35, `local.properties` mit `sdk.dir`.
Signatur: `keystore.jks` (im Repo, Passwörter in `app/build.gradle.kts`) –
**nur für Test-Builds**, für den Play-Store eigenen Keystore erzeugen!

## Architektur

```
app/src/main/java/ai/liquid/lfm3/
├── App.kt                  # Globaler Crash-Handler + Server-Start
├── MainActivity.kt         # HTML-Launcher (Systemprompt, Modelle, HTML-Eingabe, Start)
├── server/
│   └── LfmServer.kt        # NanoHTTPD: OpenAI-kompatible API + /app.html (ServerHost)
├── engine/
│   ├── ModelCatalog.kt     # Automatisch generiert aus HF tree.json (scripts/gen_catalog.py)
│   ├── ModelManager.kt     # HF-Downloads (Progress, Cancel, Resume-freundlich)
│   ├── ModelTasks.kt       # Download/Laden/Entladen-Zentrale (zustandslos überlebt UI)
│   ├── LfmTokenizer.kt     # Byte-Level-BPE (293k Merges, added_tokens, UTF-8-Streaming)
│   ├── ChatTemplate.kt     # ChatML-Template mit <think>-Handling
│   ├── OrtEngine.kt        # ONNX Runtime + QNN/HTP, KV-Cache-Zero-Copy, Sampling
│   └── EngineHost.kt       # Singleton: Laden, chatApi() für die HTTP-API
└── ui/
    ├── HtmlRunnerActivity.kt  # Vollbild-WebView (File-Upload, getUserMedia)
    ├── SysPrompt.kt           # Systemprompt-Text für externe KIs
    └── DemoHtml.kt            # Beispiel-Chat-App (Streaming-Demo)
```

Technische Details (IO-Signatur, KV-Cache-Strategie, Vision-Pipeline) sind im
Quellcode dokumentiert; die Engine liest die Tensor-Metadaten **zur Laufzeit**
aus dem ONNX-Graphen und initialisiert die Caches entsprechend – dadurch
funktionieren alle Varianten (24 Layer Conv/Attention, 11 KV-Layer) ohne
harte Shape-Annahmen.

## Modelle laden

Launcher → **„Modelle“** → Variante antippen (Download, danach automatisches
Laden; langer Tastendruck = Präzision). Das Modell wird nach
`Android/data/ai.liquid.lfm3/files/models/` geladen.

## Lizenz

Code: MIT. Modelle: siehe [LiquidAI auf Hugging Face](https://huggingface.co/LiquidAI).
