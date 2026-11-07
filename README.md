# LFM3 NPU – LiquidAI LFM3 on Hexagon NPU (Android)

Vollständige Android-App, die **alle LiquidAI LFM3 ONNX-Varianten lokal auf dem
Hexagon-NPU (Qualcomm QNN/HTP)** ausführt – inspiriert vom
[LFM Playground "Talk"](https://playground.liquid.ai/talk).

> **Hinweis:** Dies ist ein Community-Projekt und steht in keiner Verbindung zu LiquidAI.

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
- **/talk-Funktionen:** Streaming-Chat (Token-für-Token), Thinking-Modus
  (`<think>`-Stream kursiv abgesetzt), Modell-Manager mit Fortschritt/Abbruch/Löschen,
  Backend-Info-Seite, Live-Sprachmodus (SpeechRecognizer + TTS), Agent-Tab
  (JSON-Extraktion, Rechner-/Zeit-Tool-Calling, Bildanalyse)
- **Einstellungen:** System-Prompt, Thinking, Kontextlänge 1k–16k Tokens
  (mit RAM-Warnung ab 8k), Temperature, NPU-Fallback
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
├── App.kt                  # Globaler Crash-Handler
├── MainActivity.kt         # Navigation + Crash-Screen
├── engine/
│   ├── ModelCatalog.kt     # Automatisch generiert aus HF tree.json (scripts/gen_catalog.py)
│   ├── ModelManager.kt     # HF-Downloads (Progress, Cancel, Resume-freundlich)
│   ├── LfmTokenizer.kt     # Byte-Level-BPE (293k Merges, added_tokens, UTF-8-Streaming)
│   ├── ChatTemplate.kt     # ChatML-Template mit <think>-Handling
│   ├── OrtEngine.kt        # ONNX Runtime + QNN/HTP, KV-Cache-Zero-Copy, Sampling
│   └── EngineHost.kt       # Singleton: Laden, Chat, Tools
└── ui/                     # Chat / Live / Agent / Modelle / Info (Views, kein Compose)
```

Technische Details (IO-Signatur, KV-Cache-Strategie, Vision-Pipeline) sind im
Quellcode dokumentiert; die Engine liest die Tensor-Metadaten **zur Laufzeit**
aus dem ONNX-Graphen und initialisiert die Caches entsprechend – dadurch
funktionieren alle Varianten (24 Layer Conv/Attention, 11 KV-Layer) ohne
harte Shape-Annahmen.

## Modelle laden

Tab **„Modelle“** → Variante wählen → Präzision mit `q4`-Button durchschalten →
**Herunterladen** (HuggingFace, mit Fortschrittsbalken) → **Laden**.
Das Modell wird nach `Android/data/ai.liquid.lfm3/files/models/` geladen.

## Lizenz

Code: MIT. Modelle: siehe [LiquidAI auf Hugging Face](https://huggingface.co/LiquidAI).
