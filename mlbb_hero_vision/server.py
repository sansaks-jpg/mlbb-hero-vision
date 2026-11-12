"""Interactive offline Web UI tester for MLBB Hero Vision (drag-to-crop & classify)."""

from __future__ import annotations

import argparse
import base64
import io
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from .student import HeroVision, is_student_ready

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>MLBB Hero Vision — 100% Offline ONNX Tester</title>
  <style>
    :root { color-scheme: dark; --bg: #0c111d; --card: #131b2e; --border: #263352; --accent: #38bdf8; --ok: #22c55e; --text: #e2e8f0; --muted: #94a3b8; }
    * { box-sizing: border-box; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); padding: 24px; }
    .wrap { max-width: 1080px; margin: 0 auto; display: grid; gap: 18px; }
    .header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; }
    .badge { font-size: 12px; padding: 5px 12px; border-radius: 999px; background: #1e293b; border: 1px solid var(--border); color: var(--accent); }
    .grid { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 16px; display: grid; gap: 12px; align-content: start; }
    label { font-size: 13px; color: var(--muted); display: block; margin-bottom: 6px; }
    input[type="file"] { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid var(--border); background: #0b1020; color: var(--text); font-size: 14px; }
    .row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    button { padding: 10px 16px; border-radius: 10px; border: 0; background: var(--accent); color: #082f49; font-weight: 700; cursor: pointer; font-size: 14px; }
    button.secondary { background: #1e293b; color: var(--text); border: 1px solid var(--border); }
    button:disabled { opacity: 0.5; cursor: wait; }
    .canvas-wrap { position: relative; background: #070b14; border: 1px dashed var(--border); border-radius: 12px; min-height: 280px; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    canvas { max-width: 100%; max-height: 440px; cursor: crosshair; display: block; }
    .hint, .sub { font-size: 12px; color: var(--muted); }
    .hero-banner { padding: 14px; border-radius: 12px; background: #0b1326; border: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    .hero-name { font-size: 28px; font-weight: 800; }
    .conf { font-size: 16px; font-weight: 700; color: var(--ok); }
    .cand-list { display: grid; gap: 8px; }
    .cand { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-radius: 10px; background: #0b1020; border: 1px solid #1c2742; }
    pre { margin: 0; padding: 12px; border-radius: 10px; background: #070b14; border: 1px solid #1e293b; overflow: auto; font-size: 12px; max-height: 220px; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <h2 style="margin:0">MLBB Hero Vision</h2>
      <div class="hint">100% Offline ONNX Recognition (ConvNeXt-Tiny distilled from Gemini Embedding 2)</div>
    </div>
    <span class="badge" id="statsBadge">Loading...</span>
  </div>

  <div class="grid">
    <div class="card">
      <div>
        <label>1. Upload Draft Pick / Ban / HUD Screenshot (drag box on image to crop a slot)</label>
        <input type="file" id="fileInput" accept="image/*" />
      </div>
      <div class="canvas-wrap"><canvas id="previewCanvas"></canvas></div>
      <div class="row" style="justify-content:space-between">
        <span class="hint" id="cropInfo">Upload an image to begin.</span>
        <button class="secondary" id="resetCropBtn" type="button">Reset Crop</button>
      </div>
      <div class="row">
        <button id="detectBtn" type="button" disabled>Detect Hero (Offline)</button>
      </div>
    </div>

    <div class="card">
      <div class="hero-banner">
        <div>
          <div class="sub">PREDICTION</div>
          <div class="hero-name" id="finalHero">—</div>
          <div class="sub" id="finalSource">Waiting for image...</div>
        </div>
        <div style="text-align:right">
          <div class="conf" id="finalConf">—</div>
          <div class="sub" id="timingInfo"></div>
        </div>
      </div>

      <div>
        <label>Top-5 Class Probabilities</label>
        <div class="cand-list" id="candList"></div>
      </div>

      <div>
        <label>Raw JSON</label>
        <pre id="rawJson">{}</pre>
      </div>
    </div>
  </div>
</div>

<script>
const fileInput = document.getElementById('fileInput');
const canvas = document.getElementById('previewCanvas');
const ctx = canvas.getContext('2d');
const detectBtn = document.getElementById('detectBtn');
const resetCropBtn = document.getElementById('resetCropBtn');
const cropInfo = document.getElementById('cropInfo');

fetch('/api/status').then(r => r.json()).then(s => {
  document.getElementById('statsBadge').textContent =
    `Offline ONNX Ready | ${s.classes} classes | ${s.index_vectors} retrieval vectors`;
});

let loadedImg = null, dragStart = null, selection = null;

function draw() {
  if (!loadedImg) return;
  canvas.width = loadedImg.naturalWidth;
  canvas.height = loadedImg.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(loadedImg, 0, 0);
  if (selection && selection.w > 4 && selection.h > 4) {
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = Math.max(2, canvas.width / 250);
    ctx.strokeRect(selection.x, selection.y, selection.w, selection.h);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.16)';
    ctx.fillRect(selection.x, selection.y, selection.w, selection.h);
  }
}

fileInput.addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => {
    loadedImg = img;
    selection = null;
    draw();
    detectBtn.disabled = false;
    cropInfo.textContent = `Loaded ${img.naturalWidth}x${img.naturalHeight}. Drag a box or click Detect.`;
  };
  img.src = URL.createObjectURL(f);
});

function toImgCoords(evt) {
  const rect = canvas.getBoundingClientRect();
  return { x: (evt.clientX - rect.left) * (canvas.width / rect.width), y: (evt.clientY - rect.top) * (canvas.height / rect.height) };
}
canvas.addEventListener('mousedown', (e) => { if (loadedImg) dragStart = toImgCoords(e); });
canvas.addEventListener('mousemove', (e) => {
  if (!dragStart) return;
  const p = toImgCoords(e);
  selection = { x: Math.min(dragStart.x, p.x), y: Math.min(dragStart.y, p.y), w: Math.abs(p.x - dragStart.x), h: Math.abs(p.y - dragStart.y) };
  draw();
});
window.addEventListener('mouseup', () => {
  if (!dragStart) return;
  dragStart = null;
  if (selection && selection.w > 8 && selection.h > 8) {
    cropInfo.textContent = `Crop selected: ${Math.round(selection.w)}x${Math.round(selection.h)} px`;
  } else {
    selection = null; draw();
  }
});
resetCropBtn.addEventListener('click', () => { selection = null; draw(); });

detectBtn.addEventListener('click', async () => {
  if (!loadedImg) return;
  const outCanvas = document.createElement('canvas');
  if (selection && selection.w > 8 && selection.h > 8) {
    outCanvas.width = Math.round(selection.w);
    outCanvas.height = Math.round(selection.h);
    outCanvas.getContext('2d').drawImage(loadedImg, selection.x, selection.y, selection.w, selection.h, 0, 0, outCanvas.width, outCanvas.height);
  } else {
    outCanvas.width = loadedImg.naturalWidth;
    outCanvas.height = loadedImg.naturalHeight;
    outCanvas.getContext('2d').drawImage(loadedImg, 0, 0);
  }
  detectBtn.disabled = true;
  detectBtn.textContent = 'Running ONNX...';
  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: outCanvas.toDataURL('image/png') })
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Request failed');
    document.getElementById('finalHero').textContent = data.hero || 'NOT AN MLBB HERO / UNKNOWN';
    document.getElementById('finalSource').textContent = data.rejection_reason
      ? `Rejected (${data.rejection_reason})`
      : `Engine: ${data.engine}`;
    document.getElementById('finalConf').textContent = `${(data.confidence * 100).toFixed(2)}%`;
    document.getElementById('timingInfo').textContent = `${data.latency_ms} ms (CPU)`;
    const list = document.getElementById('candList');
    list.innerHTML = '';
    for (const p of (data.top_probabilities || [])) {
      const div = document.createElement('div');
      div.className = 'cand';
      div.innerHTML = `<span style="font-weight:700">${p.hero}</span><span style="color:#38bdf8;font-weight:700">${(p.prob * 100).toFixed(2)}%</span>`;
      list.appendChild(div);
    }
    document.getElementById('rawJson').textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    alert(err.message);
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = 'Detect Hero (Offline)';
  }
});
</script>
</body>
</html>
"""


def run_server(host: str = "127.0.0.1", port: int = 8787, model_dir: str | None = None) -> None:
    """Start the offline HeroVision test server."""
    vision = HeroVision(model_dir=model_dir, auto_download=True)

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/status":
                self._send_json(
                    {
                        "student_ready": is_student_ready(vision.model_dir),
                        "classes": len(vision.labels),
                        "index_vectors": int(vision.db_mat.shape[0]),
                    }
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/api/predict":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                data_url = str(body.get("image", ""))
                b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
                with Image.open(io.BytesIO(base64.b64decode(b64))) as img:
                    res = vision.predict(img, top_k=int(body.get("top_k", 5)))
                self._send_json(res)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"MLBB Hero Vision Web UI running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Start MLBB Hero Vision offline web tester")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--model-dir", default=None)
    a = p.parse_args()
    run_server(host=a.host, port=a.port, model_dir=a.model_dir)
