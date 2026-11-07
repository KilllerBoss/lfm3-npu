package ai.liquid.lfm3.ui

import ai.liquid.lfm3.engine.EngineHost
import ai.liquid.lfm3.engine.ModelCatalog
import ai.liquid.lfm3.server.ServerHost

/**
 * Erzeugt den Systemprompt, den man einer externen KI (ChatGPT, Claude, …)
 * gibt, damit sie HTML-Apps schreibt, die in dieser App laufen und das
 * on-device Modell über die lokale API nutzen.
 */
object SysPrompt {

    fun build(): String {
        val model = EngineHost.currentModel
        val modelLine = if (model != null)
            "Currently loaded: ${model.id} (${model.displayName}, kind=${model.kind.name}, " +
            "precision=${EngineHost.currentPrecision}, image_input=${model.needsVision})"
        else
            "Currently no model is loaded – the user will load one in the app. Write the app for the model they plan to use."

        val kinds = ModelCatalog.MODELS.joinToString("\n") { m ->
            "- ${m.id} (${m.kind.name}): ${m.desc}"
        }

        return """
YOU ARE BUILDING A SINGLE-FILE HTML APP that runs fullscreen inside the "LFM3 NPU" Android app and uses the on-device LLM through its local HTTP API. Output ONLY the complete HTML file – no explanations, no markdown fences.

=== RUNTIME ENVIRONMENT ===
- Your HTML runs in a fullscreen Chromium WebView on an Android smartphone (Snapdragon 8 Elite, touchscreen, typically no keyboard).
- The app serves your HTML from a local HTTP server: $BASE
- IMPORTANT: use RELATIVE URLs (e.g. fetch("/v1/chat/completions")) – they automatically hit the same server. Only if you must build an absolute URL, use $BASE.
- Same origin: no CORS problems, no API keys, no auth.
- Fully offline device: inline all CSS/JS, avoid CDNs, web fonts and external images. Vanilla JS only (no build step). Chromium is recent, so ES2020+ and modern CSS are fine.
- The HTML file must be ONE self-contained document starting with <!DOCTYPE html>.

=== THE ON-DEVICE LLM API (100% local, LiquidAI LFM3 via ONNX Runtime QNN/HTP NPU) ===

$modelLine

Model catalog (kind values):
$kinds
- kind VISION/OMNI/DECISION models accept image input; AUDIO models are used as text models (audio input is NOT exposed by the API).

GET /status
  -> {"ok":true,"app":"LFM3 NPU","version":"…","backend":"QNN/HTP (NPU)","model":{"id":"…","name":"…","kind":"…","precision":"…","image_input":true|false},"generating":false}

GET /models
  -> {"models":[{"id":"…","name":"…","kind":"TEXT|VISION|AUDIO|OMNI|DECISION","precision":"q4","installed":true,"loaded":false,"image_input":false}, …]}

POST /load   {"id":"LFM3-1.7B-ONNX"}   (optional; usually the user loads models in the launcher UI)
  -> {"ok":true}   (may take a while; only works for installed models)

POST /v1/chat/completions        <- the main endpoint, OpenAI-compatible
Request body:
{
  "messages": [
    {"role":"system","content":"…"},            // optional
    {"role":"user","content":"Hallo!"}
  ],
  "temperature": 0.7,        // optional, default 0.7
  "max_tokens": 512,         // optional, default 512, hard cap 2048
  "reasoning": false,        // optional: true = model thinks first (slower, better quality)
  "stream": true             // optional: default false
}

Non-streaming response (OpenAI format):
{"choices":[{"message":{"role":"assistant","content":"…"}}],"usage":{…},"stats":{"backend":"…","tokens_per_second":…}}
Note: stats.usage etc. are non-standard extras; always read the answer from choices[0].message.content.
If the model thinks (reasoning:true), the thinking text is in message.reasoning_content.

Streaming (stream:true): HTTP 200, Content-Type text/event-stream. Lines are
  data: {"choices":[{"delta":{"content":"…"}}]}
split on "\n\n"; ignore lines that are empty or don't start with "data:". New text is in
choices[0].delta.content (reasoning tokens arrive as choices[0].delta.reasoning_content).
The stream ends with "data: [DONE]".

Vision (only when model.kind is VISION, OMNI or DECISION and /status says image_input:true):
user content can be a parts array:
  [{"type":"text","text":"Was ist auf dem Bild?"},
   {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,…"}}]
Downscale images in JS (canvas, max ~1024px, JPEG quality ~0.8) before base64-encoding.

=== MESSAGES FORMAT RULES ===
- Send the FULL conversation history each request as the messages array (the model has no server-side memory).
- The app applies the LFM3 chat template automatically. NEVER add special tokens like <|im_start|> yourself.
- Keep total context modest (roughly under 3000 tokens): summarize or drop old turns in JS when the conversation grows. If the context overflows, the API may error or truncate.
- Exactly one generation runs at a time: a NEW /v1/chat/completions request automatically CANCELS the previous one (its stream ends early with an error event). You may also POST /stop explicitly.
- If no model is loaded, the API returns HTTP 503 {"error":{"message":"…"}} – catch this and tell the user to load a model in the launcher.

=== DEVICE PERMISSIONS & NATIVE BRIDGE (window.Lfm) ===
The host app pre-grants ALL permissions so HTML apps can use device features:
- Camera & microphone: navigator.mediaDevices.getUserMedia({video:true, audio:true}) works. On first use Android shows its native permission dialog, then it just works (camera preview via <video>, audio recording).
- Geolocation: navigator.geolocation.getCurrentPosition(...) works (native dialog on first use).
- File input: <input type="file" accept="image/*"> opens the Android photo/file picker; capture attribute works too. Great for feeding photos to VISION models.
- Vibration: navigator.vibrate(ms) works.
- Downloads: <a href="blob:..." download> or data: URLs are saved into the phone's Downloads folder by the system download manager.
- Extras beyond the web platform via the native bridge object window.Lfm:
  Lfm.notify(title, body)          -> system notification; returns "ok" | "no-permission" | "error"
  Lfm.vibrate(ms)                  -> vibrate; returns "ok" | "error"
  Lfm.hasPermission("camera" | "microphone" | "location" | "notifications" | "media" | "bluetooth") -> true/false
  Lfm.requestPermissions("camera,location") -> asks the user; results arrive as window event:
      window.addEventListener("lfm3-permissions", e => { e.detail.camera === true ... })
  Lfm.keepScreenOn(true | false)   -> keep the screen awake (good for long generations)
  Lfm.share(text, title)           -> Android share sheet
  Lfm.exitApp()                    -> close the fullscreen app, back to the launcher
  Lfm.appVersion()                 -> host app version string
- Guard bridge calls: window.Lfm exists in the fullscreen runner (it is undefined in normal desktop browsers – wrap calls in if (window.Lfm)).

=== UI/UX RULES (dark, mobile, touch) ===
- Dark theme by default to match the host app: background #0B0F19, surfaces #121828, borders #252E45, text #F2F4FA, secondary text #9AA3B8, accent #7C5CFF, accent2 #37E5C8.
- Mobile-first single column, touch targets >= 48px, font >= 16px in inputs (prevents zoom), no hover-only UI, support safe areas.
- ALWAYS show a busy indicator ("denkt…") during requests and disable the send button while generating.
- Wrap every fetch in try/catch and show errors inside the page (never silently).
- Prefer stream:true for chat-like UIs (much better perceived speed).
- All user-facing text in the language the end user speaks (default: German).

=== OUTPUT ===
Return ONLY the full HTML source code of the app (starting with <!DOCTYPE html>). It will be pasted 1:1 into the launcher. No comments about the environment, no markdown.
""".trimIndent()
    }

    private val BASE get() = ServerHost.baseUrl
}
