package ai.liquid.lfm3.ui

/**
 * Kleine Demo-HTML-App (Streaming-Chat), die zeigt, wie man die lokale
 * API benutzt. Wird über den "Beispiel"-Button ins Eingabefeld geladen.
 * Bewusst ohne Template-Literale mit ${} geschrieben (Kotlin-Raw-Strings).
 */
object DemoHtml {

    val html: String = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>LFM3 Chat-Demo</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; }
  body { background:#0B0F19; color:#F2F4FA; font-family:system-ui,sans-serif;
         display:flex; flex-direction:column; height:100dvh; }
  header { padding:14px 16px; background:#121828; border-bottom:1px solid #252E45;
           font-size:14px; color:#9AA3B8; }
  header b { color:#37E5C8; }
  #chat { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:85%; padding:10px 14px; border-radius:14px; font-size:15px; line-height:1.45; white-space:pre-wrap; }
  .user { align-self:flex-end; background:#7C5CFF; color:#fff; border-bottom-right-radius:4px; }
  .bot  { align-self:flex-start; background:#1A2236; border-bottom-left-radius:4px; }
  .err  { align-self:center; background:#3a1a1a; color:#FF6B6B; font-size:13px; }
  form { display:flex; gap:8px; padding:10px; background:#121828; border-top:1px solid #252E45;
         padding-bottom:calc(10px + env(safe-area-inset-bottom)); }
  input { flex:1; background:#1A2236; border:1px solid #252E45; color:#F2F4FA;
          border-radius:10px; padding:12px 14px; font-size:16px; outline:none; }
  input:focus { border-color:#7C5CFF; }
  button { background:#7C5CFF; color:#fff; border:0; border-radius:10px; width:52px; font-size:20px; }
  button:disabled { opacity:.4; }
</style>
</head>
<body>
<header>LFM3 NPU · <b id="model">verbinde…</b></header>
<div id="chat"></div>
<form id="f">
  <input id="in" autocomplete="off" placeholder="Frage an das Modell…">
  <button id="go">➤</button>
</form>
<script>
"use strict";
var chat = document.getElementById("chat");
var input = document.getElementById("in");
var btn = document.getElementById("go");
var hist = [];

function add(cls, text) {
  var d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

fetch("/status").then(function(r){return r.json();}).then(function(s){
  var m = s.model;
  document.getElementById("model").textContent =
    m ? (m.name + " · " + s.backend) : "kein Modell geladen";
  if (!m) add("err", "Kein Modell geladen – bitte im Launcher über 'Modelle' laden.");
}).catch(function(e){ add("err", "Server nicht erreichbar: " + e); });

document.getElementById("f").onsubmit = function(e) {
  e.preventDefault();
  var q = input.value.trim();
  if (!q || btn.disabled) return;
  input.value = "";
  btn.disabled = true;
  add("user", q);
  var bot = add("bot", "…");
  var acc = "";
  var messages = hist.slice(-8);
  messages.push({role:"user", content:q});

  fetch("/v1/chat/completions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({messages: messages, stream: true, max_tokens: 512})
  }).then(function(r) {
    if (!r.ok) return r.json().then(function(j){ throw new Error((j.error && j.error.message) || ("HTTP " + r.status)); });
    var reader = r.body.getReader();
    var dec = new TextDecoder();
    var buf = "";
    function step() {
      return reader.read().then(function(res){
        if (res.done) return finish();
        buf += dec.decode(res.value, {stream:true});
        var idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          var line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 2);
          if (line.indexOf("data:") !== 0) continue;
          var data = line.slice(5).trim();
          if (data === "[DONE]") continue;
          try {
            var j = JSON.parse(data);
            if (j.error) throw new Error(j.error.message);
            var d = j.choices && j.choices[0] && j.choices[0].delta;
            var t = d && d.content;
            if (t) { acc += t; bot.textContent = acc; chat.scrollTop = chat.scrollHeight; }
          } catch(_e) {}
        }
        return step();
      });
    }
    function finish() {
      if (!acc) bot.textContent = "(leere Antwort)";
      hist.push({role:"user", content:q}, {role:"assistant", content:acc});
      btn.disabled = false;
    }
    return step();
  }).catch(function(err) {
    bot.textContent = "";
    add("err", "Fehler: " + err.message);
    btn.disabled = false;
  });
};
</script>
</body>
</html>"""
}
