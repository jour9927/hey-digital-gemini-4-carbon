#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon V2 — Three.js WebGL2 3D Motion Graphics & Interactive UI Studio
============================================================================================
Reconstructs `@hey_digitall`'s true 3D motion-design and interactive UI choreography using
Playwright + Apple Metal WebGL2 (`Google Chrome`) + local `Three.js` (`assets/vendor/three.min.js`):

Long-Form (`1920x1080 @ 30fps`) Templates:
1. `webgl_memory_lanes`   — Sunlit `#ECEAE4` 3D studio floor (`PCFSoftShadowMap`), 2–3 metallic
                            channels with 16 bevelled 3D blocks popping in with spring physics,
                            camera dolly, morphing giant thin hero metric, and 3D-anchored HUD.
2. `webgl_stacked_tco`    — 3D aluminum laptop -> smooth cubic-bezier 3D camera pan to dual
                            pedestals where upgrade price slabs drop & bounce (`BASE vs PRO`).
3. `webgl_ssd_particles`  — 3D NVMe NAND die array filling up while 30 glowing 3D data cubes fly
                            along parabolic arcs into an open 3D aluminum laptop.
4. `webgl_chip_die`       — Cinematic orbital 3D camera over a detailed Apple/AI Silicon SoC package
                            with glowing CPU/GPU/NPU 3D core clusters and SMD capacitors.
5. `webgl_dual_laptop`    — Side-by-side 3D open laptops comparing sustained speed & thermal aura
                            (orange passive heat waves vs cool cyan active airflow rings).
6. `ui_configurator_crawl`— 3D-angled Apple-style configurator with animated macOS cursor hover/click
                            ripples + live crawled public article window with yellow highlighter sweep.

Traffic-Funnel Short (`1080x1920 @ 30fps`) Templates:
1. `short_hook_dynamic`   — Live vertical motion video (`<video>` or pan/zoom still) while 5 die-cut
                            stickers & vector icons pop in sequentially with spring bounce.
2. `short_phone_interactive` — Mint-sage `#DCEFD8` studio layout + emerald iPhone Pro bezel + 4 live
                            iOS rows with touch-cursor ripples, flipping green toggles, and yellow focus.
"""
from __future__ import annotations

import functools
import http.server
import json
import mimetypes
import os
import socketserver
import subprocess
import threading
import urllib.parse
from pathlib import Path
from typing import Optional


CHROME_EXEC = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


class _QuietStaticHandler(http.server.BaseHTTPRequestHandler):
    root_map: dict[str, Path] = {}
    dynamic_html: dict[str, bytes] = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p = urllib.parse.unquote(parsed.path)
        if p in self.dynamic_html:
            body = self.dynamic_html[p]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        target: Optional[Path] = None
        for prefix, folder in self.root_map.items():
            if p.startswith(prefix + "/"):
                rel = p[len(prefix) + 1 :]
                cand = (folder / rel).resolve()
                if str(cand).startswith(str(folder.resolve())) and cand.is_file():
                    target = cand
                    break

        if target is None:
            self.send_response(404)
            self.end_headers()
            return

        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def caption_for_timestamp(beat: dict, caption_timings: list[dict], t_local: float, real_dur: float) -> str:
    """Resolve exact spoken caption at `t_local` using Whisper-aligned `caption_timings` (or char-weighted fallback)."""
    if caption_timings:
        for seg in caption_timings:
            if float(seg["start"]) - 0.03 <= t_local <= float(seg["end"]) + 0.06:
                return seg["text"]
        if t_local < float(caption_timings[0]["start"]):
            return caption_timings[0]["text"] if t_local >= 0.04 else ""
        # Between breath-separated sentences or after final speech offset: show clean frame (NEVER flash last caption!)
        return ""

    caps = beat.get("captions") or []
    if not caps:
        return ""
    active_dur = max(0.4, real_dur - 0.22)
    if t_local > active_dur + 0.12:
        return ""
    tot = sum(max(1, len(c)) for c in caps)
    acc = 0.0
    for c in caps:
        span = active_dur * (max(1, len(c)) / tot)
        if t_local <= acc + span + 1e-4:
            return c
        acc += span
    return caps[-1]


COMMON_LONG_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; -webkit-font-smoothing: antialiased; }
  html, body {
    width: 1920px; height: 1080px; overflow: hidden;
    background: radial-gradient(circle at 52% 44%, #F6F4EF 0%, #EAE7DF 68%, #DFDCD3 100%);
    font-family: "PingFang TC", "SF Pro Display", "Helvetica Neue", sans-serif;
    color: #18181A;
  }
  .studio-grid {
    position: absolute; inset: 0; pointer-events: none; z-index: 1;
    background-image: radial-gradient(rgba(80, 76, 68, 0.12) 1.2px, transparent 1.2px);
    background-size: 36px 36px;
  }
  #gl-wrap { position: absolute; inset: 0; z-index: 2; }
  #hud { position: absolute; inset: 0; z-index: 5; pointer-events: none; }

  .chapter-hud {
    position: absolute; top: 54px; right: 78px;
    font-size: 23px; font-weight: 500; color: #5C5A54; letter-spacing: 0.04em;
    background: rgba(255,255,255,0.62); backdrop-filter: blur(8px);
    padding: 8px 20px; border-radius: 999px; border: 1px solid rgba(160,155,145,0.28);
  }
  .hero-box {
    position: absolute; top: 58px; left: 96px;
    display: flex; flex-direction: column; gap: 4px;
  }
  .hero-row { display: flex; align-items: baseline; gap: 16px; }
  .hero-num {
    font-family: "Helvetica Neue", "SF Pro Display", "PingFang TC", sans-serif;
    font-weight: 200; font-size: 162px; line-height: 0.96; letter-spacing: -0.03em;
  }
  .hero-unit {
    font-family: "Helvetica Neue", "SF Pro Display", "PingFang TC", sans-serif;
    font-weight: 300; font-size: 52px; color: #66635B;
  }
  .hero-delta {
    display: inline-flex; align-items: center; padding: 6px 18px; border-radius: 999px;
    font-size: 28px; font-weight: 700; color: #FFF; background: #D95D1E;
    box-shadow: 0 8px 20px rgba(217, 93, 30, 0.25);
    transform-origin: left center;
  }
  .hero-sub {
    font-size: 23px; font-weight: 500; color: #706D65; margin-top: 4px; max-width: 780px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }

  .tag-3d {
    position: absolute; transform: translate(-50%, -50%);
    background: rgba(255, 255, 255, 0.90); border: 1px solid rgba(180, 175, 165, 0.55);
    box-shadow: 0 8px 24px rgba(28, 26, 22, 0.09);
    padding: 8px 18px; border-radius: 14px; font-size: 24px; font-weight: 600; color: #22211F;
    white-space: nowrap;
  }
  .tag-3d .val { font-weight: 700; margin-left: 8px; }

  .source-pill {
    position: absolute; left: 96px; bottom: 136px;
    display: inline-flex; align-items: center; gap: 10px;
    background: rgba(255, 255, 255, 0.84); border: 1px solid rgba(196, 191, 180, 0.85);
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    padding: 8px 20px; border-radius: 999px; font-size: 20px; color: #4A4843; font-weight: 500;
  }
  .source-dot { width: 10px; height: 10px; border-radius: 50%; background: #14804A; }

  .caption-bar {
    position: absolute; left: 50%; bottom: 48px; transform: translateX(-50%);
    font-size: 40px; font-weight: 600; color: #171719;
    text-shadow: 0 1px 0 rgba(255,255,255,0.75), 0 4px 16px rgba(255,255,255,0.45);
    letter-spacing: 0.02em; text-align: center; white-space: nowrap;
  }

  /* UI Configurator + Crawled Article Spotlight */
  .ui-stage {
    position: absolute; inset: 0; z-index: 3;
    display: flex; align-items: center; justify-content: center; gap: 46px;
    padding: 150px 96px 165px 96px; perspective: 1500px;
  }
  .cfg-card {
    width: 760px; background: rgba(255,255,255,0.95); border-radius: 28px;
    border: 1px solid rgba(190,186,176,0.7);
    box-shadow: 0 28px 60px rgba(24,22,18,0.11);
    padding: 34px 38px; transform: rotateY(5deg) rotateX(2deg);
  }
  .cfg-eyebrow { font-size: 19px; font-weight: 700; color: #D96B27; letter-spacing: 0.08em; text-transform: uppercase; }
  .cfg-title { font-size: 34px; font-weight: 700; color: #161618; margin: 6px 0 22px 0; display: flex; justify-content: space-between; align-items: baseline; }
  .cfg-price { font-family: "Helvetica Neue", sans-serif; font-size: 40px; font-weight: 700; color: #0E6EB8; }
  .cfg-opt {
    display: flex; justify-content: space-between; align-items: center;
    padding: 18px 22px; border-radius: 18px; border: 2px solid #E3E0D8;
    margin-bottom: 14px; background: #FAF9F6; transition: none;
  }
  .cfg-opt.active {
    border-color: #0E6EB8; background: #F0F7FD;
    box-shadow: 0 8px 22px rgba(14,110,184,0.12);
  }
  .cfg-opt-name { font-size: 24px; font-weight: 600; color: #1E1E20; }
  .cfg-opt-sub { font-size: 18px; color: #6E6C66; margin-top: 3px; }
  .cfg-opt-delta { font-size: 23px; font-weight: 700; color: #3A3935; }
  .cfg-opt.active .cfg-opt-delta { color: #0E6EB8; }

  .browser-card {
    width: 860px; background: #161920; color: #F4F6FA; border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 32px 70px rgba(12,14,20,0.24);
    overflow: hidden; transform: rotateY(-6deg) rotateX(2deg);
  }
  .browser-top {
    height: 54px; background: #20242E; display: flex; align-items: center; padding: 0 22px; gap: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }
  .b-dot { width: 13px; height: 13px; border-radius: 50%; }
  .b-url {
    margin-left: 12px; background: #12151B; color: #A6B0C0; font-size: 17px;
    padding: 6px 18px; border-radius: 10px; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .browser-body { padding: 32px 36px; }
  .pub-row { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
  .pub-badge { background: #E59934; color: #121418; font-weight: 700; font-size: 17px; padding: 4px 12px; border-radius: 8px; }
  .pub-date { color: #94A0B4; font-size: 18px; }
  .art-title { font-size: 29px; font-weight: 700; line-height: 1.34; color: #FFFFFF; margin-bottom: 22px; }
  .quote-box {
    background: rgba(255,255,255,0.05); border-left: 4px solid #F0CA86;
    padding: 18px 22px; border-radius: 0 16px 16px 0; margin-bottom: 20px;
    font-size: 24px; line-height: 1.5; color: #EAEFF7;
  }
  .hl-sweep {
    background-image: linear-gradient(120deg, rgba(240, 202, 134, 0.36) 0%, rgba(229, 153, 52, 0.36) 100%);
    background-repeat: no-repeat; background-position: 0 88%;
    padding: 2px 4px; border-radius: 4px;
  }
  .metric-chips { display: flex; flex-wrap: wrap; gap: 10px; }
  .m-chip {
    background: rgba(14, 110, 184, 0.25); border: 1px solid rgba(86, 172, 242, 0.4);
    color: #9BD2FF; font-size: 19px; font-weight: 600; padding: 6px 14px; border-radius: 999px;
  }

  .cursor-ptr {
    position: absolute; z-index: 20; width: 38px; height: 38px; pointer-events: none;
    filter: drop-shadow(0 6px 12px rgba(0,0,0,0.28));
  }
  .click-ripple {
    position: absolute; z-index: 19; width: 64px; height: 64px; border-radius: 50%;
    border: 3px solid #0E6EB8; transform: translate(-50%, -50%) scale(0); opacity: 0; pointer-events: none;
  }
</style>
<script src="__THREE_URL__"></script>
</head>
<body>
<div class="studio-grid"></div>
<div id="gl-wrap"></div>
<div id="ui-wrap"></div>
<div id="hud">
  <div class="chapter-hud" id="ch-hud"></div>
  <div class="hero-box" id="hero-box">
    <div class="hero-row">
      <div class="hero-num" id="hero-num"></div>
      <div class="hero-unit" id="hero-unit"></div>
      <div class="hero-delta" id="hero-delta" style="display:none"></div>
    </div>
    <div class="hero-sub" id="hero-sub"></div>
  </div>
  <div id="labels-3d"></div>
  <div class="source-pill" id="src-pill"><span class="source-dot"></span><span id="src-txt"></span></div>
  <div class="caption-bar" id="cap-bar"></div>
</div>

<script>
window.BEAT = __BEAT_JSON__;
window.__READY = false;

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function easeOutCubic(x) { x = clamp(x, 0, 1); return 1 - Math.pow(1 - x, 3); }
function easeInOutCubic(x) { x = clamp(x, 0, 1); return x < 0.5 ? 4*x*x*x : 1 - Math.pow(-2*x + 2, 3)/2; }
function easeOutBack(x) {
  x = clamp(x, 0, 1);
  const c1 = 1.70158, c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
}

const B = window.BEAT;
const accentStr = B.accent_hex || "#0E6EB8";
const accentInt = parseInt(accentStr.replace("#", ""), 16) || 0x0E6EB8;

document.getElementById("ch-hud").textContent = `${String(B.chapter_num || "01").padStart(2,"0")}  ·  ${B.chapter_title || "核心拆解"}`;
document.getElementById("src-txt").textContent = `${B.source_badge || "公開技術規格"} · ${B.source_id || "SRC-01"}`;

const heroNumEl = document.getElementById("hero-num");
const heroUnitEl = document.getElementById("hero-unit");
const heroDeltaEl = document.getElementById("hero-delta");
const heroSubEl = document.getElementById("hero-sub");
heroNumEl.style.color = accentStr;
heroUnitEl.textContent = B.hero_unit || "";
const srcDet = B.source_detail || {};
heroSubEl.textContent = srcDet.title ? `${srcDet.publisher || ""}：${srcDet.title.slice(0, 42)}` : (B.chapter_title || "");

const rawHeroStr = String(B.hero_metric || "");
const hVisualLen = Array.from(rawHeroStr).reduce((acc, ch) => acc + (ch.charCodeAt(0) < 128 ? 0.62 : 1.0), 0);
heroNumEl.style.fontSize = hVisualLen <= 4.2 ? "148px" : (hVisualLen <= 7.5 ? "98px" : "66px");
const numMatch = rawHeroStr.match(/-?\d+(\.\d+)?/);
const heroTargetVal = numMatch ? parseFloat(numMatch[0]) : null;
const heroPrefix = numMatch ? rawHeroStr.slice(0, numMatch.index) : "";
const heroSuffix = numMatch ? rawHeroStr.slice(numMatch.index + numMatch[0].length) : "";
const heroDecimals = (numMatch && numMatch[0].includes(".")) ? numMatch[0].split(".")[1].length : 0;

if (!rawHeroStr || B.visual_type === "ui_configurator_crawl") {
  document.getElementById("hero-box").style.display = "none";
}

function updateHeroMetric(t) {
  if (!rawHeroStr || B.visual_type === "ui_configurator_crawl") return;
  if (heroTargetVal !== null) {
    const p = easeOutCubic(t / 1.05);
    const cur = (heroTargetVal * (0.15 + 0.85 * p)).toFixed(heroDecimals);
    heroNumEl.textContent = heroPrefix + cur + heroSuffix;
  } else {
    heroNumEl.textContent = rawHeroStr;
  }
  if (B.secondary_metric) {
    heroDeltaEl.style.display = "inline-flex";
    heroDeltaEl.textContent = B.secondary_metric;
    const sp = easeOutBack((t - 1.25) / 0.45);
    heroDeltaEl.style.opacity = clamp((t - 1.25) / 0.25, 0, 1);
    heroDeltaEl.style.transform = `scale(${Math.max(0.01, sp)})`;
  }
}

let scene, camera, renderer;
const labelsContainer = document.getElementById("labels-3d");
const trackedLabels = [];

function add3DLabel(html, worldVec) {
  const el = document.createElement("div");
  el.className = "tag-3d";
  el.innerHTML = html;
  labelsContainer.appendChild(el);
  trackedLabels.push({ el, vec: worldVec });
  return el;
}

function sync3DLabels() {
  if (!camera) return;
  const minSy = (rawHeroStr && B.visual_type !== "ui_configurator_crawl") ? 304 : 136;
  for (const item of trackedLabels) {
    const v = item.vec.clone().project(camera);
    const sx = clamp((v.x * 0.5 + 0.5) * 1920, 155, 1745);
    const sy = clamp((-v.y * 0.5 + 0.5) * 1080, minSy, 815);
    item.el.style.left = `${sx.toFixed(1)}px`;
    item.el.style.top = `${sy.toFixed(1)}px`;
  }
}

function buildLaptop3D( THREE, colorHex, screenAccentHex ) {
  const grp = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color: colorHex, metalness: 0.62, roughness: 0.28 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x181A1F, metalness: 0.2, roughness: 0.55 });

  const base = new THREE.Mesh(new THREE.BoxGeometry(3.1, 0.11, 2.15), bodyMat);
  base.position.y = 0.06;
  base.castShadow = true; base.receiveShadow = true;
  grp.add(base);

  const kb = new THREE.Mesh(new THREE.BoxGeometry(2.68, 0.02, 1.06), darkMat);
  kb.position.set(0, 0.12, -0.22);
  grp.add(kb);

  const tp = new THREE.Mesh(
    new THREE.BoxGeometry(0.92, 0.015, 0.58),
    new THREE.MeshStandardMaterial({ color: colorHex, metalness: 0.5, roughness: 0.38 })
  );
  tp.position.set(0, 0.118, 0.64);
  grp.add(tp);

  const pivot = new THREE.Group();
  pivot.position.set(0, 0.11, -1.04);
  const lid = new THREE.Mesh(new THREE.BoxGeometry(3.1, 2.02, 0.07), bodyMat);
  lid.position.set(0, 1.01, 0);
  lid.castShadow = true;
  pivot.add(lid);

  const bezel = new THREE.Mesh(new THREE.BoxGeometry(2.96, 1.88, 0.02), new THREE.MeshStandardMaterial({ color: 0x0D0F14, roughness: 0.2 }));
  bezel.position.set(0, 1.01, 0.036);
  pivot.add(bezel);

  const scrMat = new THREE.MeshStandardMaterial({
    color: screenAccentHex, emissive: screenAccentHex, emissiveIntensity: 0.42, roughness: 0.18
  });
  const scr = new THREE.Mesh(new THREE.BoxGeometry(2.78, 1.66, 0.022), scrMat);
  scr.position.set(0, 1.01, 0.042);
  pivot.add(scr);

  pivot.rotation.x = -0.26;
  grp.add(pivot);
  grp.userData = { pivot, scr, scrMat, base, bodyMat };
  return grp;
}

let updateSceneFn = function(t, dur) {};

function initWebGL() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(34, 1920 / 1080, 0.1, 100);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setSize(1920, 1080);
  renderer.setPixelRatio(1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.getElementById("gl-wrap").appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.82));
  const hemi = new THREE.HemisphereLight(0xfffdf8, 0xd4cfc4, 0.55);
  scene.add(hemi);

  const dir = new THREE.DirectionalLight(0xfff8eb, 1.45);
  dir.position.set(8.5, 15.0, 10.5);
  dir.castShadow = true;
  dir.shadow.mapSize.width = 2048;
  dir.shadow.mapSize.height = 2048;
  dir.shadow.camera.near = 1;
  dir.shadow.camera.far = 40;
  const d = 10;
  dir.shadow.camera.left = -d; dir.shadow.camera.right = d;
  dir.shadow.camera.top = d; dir.shadow.camera.bottom = -d;
  dir.shadow.bias = -0.0005;
  scene.add(dir);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.ShadowMaterial({ opacity: 0.14 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const vtype = B.visual_type;

  if (vtype === "webgl_memory_lanes") {
    let lanes = B.lanes && B.lanes.length >= 2 ? B.lanes.slice(0, 3) : [
      { label: B.product_b || "高階滿血版", value: 273, display: (B.hero_metric || "273") + " " + (B.hero_unit || "GB/s") },
      { label: B.product_a || "標準入門版", value: 153, display: "153 GB/s" }
    ];
    const maxVal = Math.max(...lanes.map(l => parseFloat(l.value) || 100), 1);
    const cols = [accentInt, 0xD96B27, 0x15804A];
    const totalBlocks = 16;
    const laneObjs = [];
    const spacingZ = lanes.length === 2 ? 2.15 : 1.65;
    const startZ = -((lanes.length - 1) * spacingZ) / 2 + 0.55;

    lanes.forEach((ln, li) => {
      const z = startZ + li * spacingZ;
      const rail = new THREE.Mesh(
        new THREE.BoxGeometry(8.6, 0.10, 0.86),
        new THREE.MeshStandardMaterial({ color: 0xD5D1C7, roughness: 0.55, metalness: 0.15 })
      );
      rail.position.set(0.3, 0.05, z);
      rail.receiveShadow = true;
      scene.add(rail);

      const activeCount = Math.max(2, Math.round(( (parseFloat(ln.value) || 60) / maxVal ) * totalBlocks));
      const blocks = [];
      for (let bi = 0; bi < totalBlocks; bi++) {
        const isAct = bi < activeCount;
        const mat = new THREE.MeshStandardMaterial({
          color: isAct ? cols[li % cols.length] : 0xCBC7BD,
          roughness: isAct ? 0.25 : 0.65,
          metalness: isAct ? 0.22 : 0.05,
          emissive: isAct ? cols[li % cols.length] : 0x000000,
          emissiveIntensity: 0.0
        });
        const box = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.56, 0.64), mat);
        const bx = -3.45 + bi * 0.50;
        box.position.set(bx, 0.36, z);
        box.castShadow = true; box.receiveShadow = true;
        scene.add(box);
        blocks.push({ box, mat, isAct, bi });
      }
      const hexStr = "#" + cols[li % cols.length].toString(16).padStart(6, "0");
      add3DLabel(`<span>${ln.label || ("組別 " + (li+1))}</span>`, new THREE.Vector3(-4.9, 0.45, z));
      add3DLabel(`<span class="val" style="color:${hexStr}">${ln.display || ln.value}</span>`, new THREE.Vector3(5.35, 0.45, z));
      laneObjs.push({ blocks, activeCount });
    });

    updateSceneFn = (t, dur) => {
      const u = easeInOutCubic(t / Math.max(3.5, dur));
      camera.position.set(1.1 - 1.9 * u, 5.8 - 0.45 * u, 8.8 - 0.7 * u);
      camera.lookAt(0.25, 0.35, 0.45);
      laneObjs.forEach((lo, li) => {
        lo.blocks.forEach(({ box, mat, isAct, bi }) => {
          const delay = li * 0.18 + bi * 0.055;
          const p = easeOutBack((t - delay) / 0.42);
          const sc = clamp(p, 0.001, 1.25);
          box.scale.set(1, sc, 1);
          box.position.y = 0.08 + 0.28 * sc;
          if (isAct) {
            const wave = 0.5 + 0.5 * Math.sin(t * 4.6 - bi * 0.45 - li);
            mat.emissiveIntensity = 0.06 + 0.18 * wave;
          }
        });
      });
    };
  }
  else if (vtype === "webgl_stacked_tco") {
    const lap = buildLaptop3D(THREE, 0xD8DADF, accentInt);
    lap.position.set(-3.55, 0, 0.25);
    lap.rotation.y = 0.38;
    scene.add(lap);

    const layersL = (B.stack_left_layers && B.stack_left_layers.length) ? B.stack_left_layers : [
      { name: "入門基礎售價", val: 999 },
      { name: "+ 升級記憶體", val: 200 },
      { name: "+ 升級 1TB SSD", val: 200 }
    ];
    const rightVal = parseFloat(B.stack_right_val) || 1599;
    const sumLeft = layersL.reduce((a, x) => a + (parseFloat(x.val) || 200), 0);
    const scaleH = 2.35 / Math.max(sumLeft, rightVal, 1);

    const slabColors = [0x0E6EB8, 0xE06D22, 0xF09A36, 0xD94826];
    const slabs = [];
    let curY = 0.06;
    layersL.forEach((ly, idx) => {
      const h = Math.max(0.36, (parseFloat(ly.val) || 200) * scaleH);
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(2.05, h, 1.75),
        new THREE.MeshStandardMaterial({ color: slabColors[idx % slabColors.length], roughness: 0.26, metalness: 0.18 })
      );
      const targetY = curY + h / 2;
      curY += h + 0.05;
      m.castShadow = true; m.receiveShadow = true;
      scene.add(m);
      slabs.push({ mesh: m, targetY, h, idx, name: ly.name, val: ly.val });
    });

    const proH = Math.max(0.9, rightVal * scaleH);
    const proBlock = new THREE.Mesh(
      new THREE.BoxGeometry(2.05, proH, 1.75),
      new THREE.MeshStandardMaterial({ color: 0x262A33, roughness: 0.28, metalness: 0.42 })
    );
    proBlock.castShadow = true; proBlock.receiveShadow = true;
    scene.add(proBlock);

    add3DLabel(`<span>${B.stack_left_label || B.product_a || "入門款加配"}</span><span class="val" style="color:#E06D22">$${sumLeft}</span>`, new THREE.Vector3(0.45, curY + 0.48, 0.2));
    add3DLabel(`<span>${B.stack_right_label || B.product_b || "Pro 高階款"}</span><span class="val" style="color:#0E6EB8">$${rightVal}</span>`, new THREE.Vector3(3.35, proH + 0.55, 0.2));

    updateSceneFn = (t, dur) => {
      const pan = easeInOutCubic((t - 0.25) / 1.6);
      camera.position.set(-1.9 + 2.7 * pan, 3.8 + 0.7 * pan, 7.2 + 1.0 * pan);
      camera.lookAt(-1.8 + 2.6 * pan, 1.15, 0.15);
      lap.rotation.y = 0.36 + Math.sin(t * 1.2) * 0.06;

      const pPro = easeOutBack((t - 0.75) / 0.65);
      proBlock.scale.y = clamp(pPro, 0.001, 1.15);
      proBlock.position.set(3.35, 0.06 + (proH * proBlock.scale.y) / 2, 0.2);

      slabs.forEach((s) => {
        if (s.idx === 0) {
          const p0 = easeOutBack((t - 0.65) / 0.55);
          const sy = clamp(p0, 0.001, 1.15);
          s.mesh.scale.y = sy;
          s.mesh.position.set(0.45, 0.06 + (s.h * sy) / 2, 0.2);
        } else {
          const tStart = 1.15 + (s.idx - 1) * 0.55;
          const pDrop = easeOutBack((t - tStart) / 0.52);
          const vis = t >= tStart - 0.05;
          s.mesh.visible = vis;
          if (vis) {
            const yNow = s.targetY + (1 - pDrop) * 2.8;
            s.mesh.position.set(0.45, Math.max(s.targetY - 0.08, yNow), 0.2);
          }
        }
      });
    };
  }
  else if (vtype === "webgl_ssd_particles") {
    const pcb = new THREE.Group();
    const board = new THREE.Mesh(
      new THREE.BoxGeometry(2.7, 0.12, 3.4),
      new THREE.MeshStandardMaterial({ color: 0x162229, roughness: 0.42, metalness: 0.35 })
    );
    board.castShadow = true; board.receiveShadow = true;
    pcb.add(board);

    const cells = [];
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const cm = new THREE.MeshStandardMaterial({ color: 0x0E6EB8, roughness: 0.24, metalness: 0.25, emissive: 0x0E6EB8, emissiveIntensity: 0.15 });
        const cell = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.28, 0.56), cm);
        cell.position.set(-0.82 + c * 0.55, 0.20, -1.05 + r * 0.70);
        cell.castShadow = true;
        pcb.add(cell);
        cells.push({ cell, cm, idx: r * 4 + c });
      }
    }
    pcb.position.set(-3.1, 0.22, 0.25);
    pcb.rotation.z = -0.10;
    pcb.rotation.y = 0.22;
    scene.add(pcb);

    const lap = buildLaptop3D(THREE, 0xDCDEE3, accentInt);
    lap.position.set(2.85, 0, 0.25);
    lap.rotation.y = -0.45;
    scene.add(lap);

    const cubes = [];
    for (let i = 0; i < 28; i++) {
      const col = i % 3 === 0 ? 0xE06D22 : 0x0E6EB8;
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(0.19, 0.19, 0.19),
        new THREE.MeshStandardMaterial({ color: col, emissive: col, emissiveIntensity: 0.35, roughness: 0.2 })
      );
      m.castShadow = true;
      scene.add(m);
      cubes.push({ mesh: m, phase: i / 28, zOff: ((i % 5) - 2) * 0.26 });
    }

    add3DLabel(`<span>NAND 快閃陣列</span>`, new THREE.Vector3(-3.1, 1.45, 0.25));
    add3DLabel(`<span>高速統一記憶體 / SSD 吞吐</span>`, new THREE.Vector3(2.85, 2.55, 0.25));

    updateSceneFn = (t, dur) => {
      const u = easeInOutCubic(t / Math.max(4, dur));
      camera.position.set(-0.5 + 1.1 * u, 4.9 - 0.3 * u, 8.3 - 0.5 * u);
      camera.lookAt(0.0, 0.9, 0.2);

      cells.forEach(({ cell, cm, idx }) => {
        const p = easeOutBack((t - idx * 0.07) / 0.45);
        const h = 0.4 + 0.85 * clamp(p, 0.1, 1.2);
        cell.scale.y = h;
        cm.emissiveIntensity = 0.12 + 0.22 * (0.5 + 0.5 * Math.sin(t * 4.0 + idx));
      });

      cubes.forEach(({ mesh, phase, zOff }) => {
        const uFly = (t * 0.58 + phase) % 1.0;
        const x = -2.6 + 5.2 * uFly;
        const y = 0.55 + 2.15 * 4 * uFly * (1 - uFly);
        mesh.position.set(x, y, 0.25 + zOff * (1 - uFly * 0.5));
        mesh.rotation.x = t * 3.2 + phase * 6;
        mesh.rotation.y = t * 2.5 + phase * 4;
      });
    };
  }
  else if (vtype === "webgl_chip_die") {
    const chip = new THREE.Group();
    const sub = new THREE.Mesh(
      new THREE.BoxGeometry(4.5, 0.18, 4.5),
      new THREE.MeshStandardMaterial({ color: 0x1E352F, roughness: 0.48, metalness: 0.28 })
    );
    sub.position.y = 0.09;
    sub.castShadow = true; sub.receiveShadow = true;
    chip.add(sub);

    const rim = new THREE.Mesh(
      new THREE.BoxGeometry(4.15, 0.22, 4.15),
      new THREE.MeshStandardMaterial({ color: 0xCFD4DC, roughness: 0.25, metalness: 0.78 })
    );
    rim.position.y = 0.14;
    chip.add(rim);

    const die = new THREE.Mesh(
      new THREE.BoxGeometry(3.05, 0.30, 3.05),
      new THREE.MeshStandardMaterial({ color: 0x14171E, roughness: 0.16, metalness: 0.65 })
    );
    die.position.y = 0.24;
    die.castShadow = true;
    chip.add(die);

    const cores = [];
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        const isGpu = r >= 3;
        const cCol = isGpu ? accentInt : 0xE06D22;
        const mat = new THREE.MeshStandardMaterial({
          color: cCol, emissive: cCol, emissiveIntensity: 0.25, roughness: 0.22, metalness: 0.4
        });
        const tile = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.14, 0.36), mat);
        tile.position.set(-1.12 + c * 0.45, 0.42, -1.12 + r * 0.45);
        chip.add(tile);
        cores.push({ tile, mat, r, c });
      }
    }
    chip.position.set(0.35, 0, 0.35);
    scene.add(chip);

    add3DLabel(`<span>晶片架構</span><span class="val" style="color:${accentStr}">${B.chip_label || "SoC 旗艦核心"}</span>`, new THREE.Vector3(0.35, 1.35, 0.35));

    updateSceneFn = (t, dur) => {
      const ang = -0.52 + 1.04 * easeInOutCubic(t / Math.max(4, dur));
      camera.position.set(0.35 + Math.sin(ang) * 6.8, 4.55 - 0.55 * Math.sin(t * 0.5), 0.35 + Math.cos(ang) * 6.8);
      camera.lookAt(0.35, 0.32, 0.35);
      cores.forEach(({ tile, mat, r, c }) => {
        const w = 0.5 + 0.5 * Math.sin(t * 5.0 - (r + c) * 0.55);
        tile.scale.y = 0.7 + 0.9 * w;
        mat.emissiveIntensity = 0.12 + 0.48 * w;
      });
    };
  }
  else if (vtype === "webgl_dual_laptop") {
    const isThermal = (B.mode || "thermal") === "thermal";
    const lapL = buildLaptop3D(THREE, 0xD9DCE0, isThermal ? 0xEA580C : 0x64748B);
    const lapR = buildLaptop3D(THREE, 0x2B2F37, 0x0E6EB8);
    lapL.position.set(-2.35, 0, 0.35); lapL.rotation.y = 0.20;
    lapR.position.set(2.35, 0, 0.35);  lapR.rotation.y = -0.20;
    scene.add(lapL); scene.add(lapR);

    const ringGeo = new THREE.RingGeometry(1.35, 1.55, 48);
    const ringMatL = new THREE.MeshBasicMaterial({ color: 0xEA580C, side: THREE.DoubleSide, transparent: true, opacity: 0.45 });
    const ringMatR = new THREE.MeshBasicMaterial({ color: 0x0EA5E9, side: THREE.DoubleSide, transparent: true, opacity: 0.45 });
    const ringL = new THREE.Mesh(ringGeo, ringMatL);
    const ringR = new THREE.Mesh(ringGeo, ringMatR);
    ringL.rotation.x = -Math.PI / 2; ringL.position.set(-2.35, 0.03, 0.35);
    ringR.rotation.x = -Math.PI / 2; ringR.position.set(2.35, 0.03, 0.35);
    scene.add(ringL); scene.add(ringR);

    const barL = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.22, 0.22), new THREE.MeshStandardMaterial({ color: isThermal ? 0xEA580C : 0x8A8F99, roughness: 0.3 }));
    const barR = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.22, 0.22), new THREE.MeshStandardMaterial({ color: 0x0E6EB8, roughness: 0.25 }));
    scene.add(barL); scene.add(barR);

    add3DLabel(`<span>${B.left_label || B.product_a || "無風扇輕薄款"}</span>`, new THREE.Vector3(-2.35, 2.28, 0.35));
    add3DLabel(`<span>${B.right_label || B.product_b || "主動散熱 Pro 款"}</span>`, new THREE.Vector3(2.35, 2.28, 0.35));

    updateSceneFn = (t, dur) => {
      const u = easeInOutCubic(t / Math.max(4, dur));
      camera.position.set(-0.6 + 1.2 * u, 3.85, 8.2 - 0.5 * u);
      camera.lookAt(0, 1.05, 0.2);

      const pulse = (t * 0.85) % 1.0;
      ringL.scale.setScalar(0.85 + 0.65 * pulse);
      ringMatL.opacity = (1 - pulse) * 0.55;
      ringR.scale.setScalar(0.85 + 0.55 * ((pulse + 0.5) % 1.0));
      ringMatR.opacity = (1 - ((pulse + 0.5) % 1.0)) * 0.55;

      const pL = clamp(easeOutCubic(t / 1.8) * 0.62, 0.05, 0.62);
      const pR = clamp(easeOutCubic(t / 1.5) * 1.00, 0.05, 1.00);
      barL.scale.x = pL; barL.position.set(-2.35 - 1.1 * (1 - pL), 1.96, 0.5);
      barR.scale.x = pR; barR.position.set(2.35 - 1.1 * (1 - pR), 1.96, 0.5);
    };
  }
}

function initUIConfigurator() {
  const wrap = document.getElementById("ui-wrap");
  const src = B.source_detail || {};
  const metrics = (src.extracted_metrics || []).slice(0, 4);
  const quote = B.highlight_quote || (src.summary || B.narration || "").slice(0, 44);
  const pA = B.product_a || "入門基礎配置";
  const pB = B.product_b || "Pro 進階配置";

  wrap.innerHTML = `
    <div class="ui-stage">
      <div class="cfg-card" id="cfg-card">
        <div class="cfg-eyebrow">HARDWARE CONFIGURATOR · 選配階梯模擬</div>
        <div class="cfg-title">
          <span>${pA} 升級試算</span>
          <span class="cfg-price" id="cfg-price">$1,099</span>
        </div>
        <div class="cfg-opt active" id="opt-0">
          <div><div class="cfg-opt-name">16GB + 512GB 起步配置</div><div class="cfg-opt-sub">日常文書瀏覽、輕量影音串流</div></div>
          <div class="cfg-opt-delta">基礎售價</div>
        </div>
        <div class="cfg-opt" id="opt-1">
          <div><div class="cfg-opt-name">升至 24GB 統一記憶體</div><div class="cfg-opt-sub">多工常駐與本機 AI 模型進階門檻</div></div>
          <div class="cfg-opt-delta">+$200</div>
        </div>
        <div class="cfg-opt" id="opt-2">
          <div><div class="cfg-opt-name">升至 1TB 高速固態硬碟</div><div class="cfg-opt-sub">總價已逼近 ${pB} 起步門檻！</div></div>
          <div class="cfg-opt-delta">+$200</div>
        </div>
      </div>

      <div class="browser-card">
        <div class="browser-top">
          <div class="b-dot" style="background:#FF5F56"></div>
          <div class="b-dot" style="background:#FFBD2E"></div>
          <div class="b-dot" style="background:#27C93F"></div>
          <div class="b-url">https://${src.domain || "9to5mac.com"}/${(src.source_id || "src-01").toLowerCase()}</div>
        </div>
        <div class="browser-body">
          <div class="pub-row">
            <span class="pub-badge">${src.publisher || "公開實測報告"}</span>
            <span class="pub-date">${src.published || "2026"} · 公開驗證編號 ${B.source_id || "SRC-01"}</span>
          </div>
          <div class="art-title">${(src.title || B.chapter_title || "").slice(0, 62)}</div>
          <div class="quote-box">「<span class="hl-sweep" id="hl-quote">${quote}</span>」</div>
          <div class="metric-chips">
            ${(metrics.length ? metrics : [B.hero_metric ? (B.hero_metric + " " + (B.hero_unit||"")) : "實測數據對照", "公開規格查證"]).map(m => `<span class="m-chip">${m}</span>`).join("")}
          </div>
        </div>
      </div>
    </div>
    <svg class="cursor-ptr" id="mac-cur" viewBox="0 0 32 32">
      <path d="M5 3 L5 26 L11.5 20 L16.2 29 L20 27 L15.2 18 L24 18 Z" fill="#111" stroke="#FFF" stroke-width="2.2" stroke-linejoin="round"/>
    </svg>
    <div class="click-ripple" id="mac-rip"></div>
  `;

  const cur = document.getElementById("mac-cur");
  const rip = document.getElementById("mac-rip");
  const priceEl = document.getElementById("cfg-price");
  const hlEl = document.getElementById("hl-quote");

  updateSceneFn = (t, dur) => {
    // Cursor glides from opt-0 -> opt-1 (click at t=1.05s) -> opt-2 (click at t=2.35s)
    let cx = 560, cy = 475;
    if (t < 1.05) {
      const p = easeInOutCubic(t / 1.05);
      cx = 720 - 150 * p; cy = 430 + 135 * p;
    } else if (t < 2.35) {
      const p = easeInOutCubic((t - 1.05) / 1.30);
      cx = 570 + 10 * p; cy = 565 + 112 * p;
    } else {
      const p = easeInOutCubic((t - 2.35) / 1.40);
      cx = 580 + 620 * p; cy = 677 - 95 * p;
    }
    cur.style.left = `${cx}px`; cur.style.top = `${cy}px`;

    const c1 = t >= 1.05;
    const c2 = t >= 2.35;
    document.getElementById("opt-1").className = "cfg-opt" + (c1 ? " active" : "");
    document.getElementById("opt-2").className = "cfg-opt" + (c2 ? " active" : "");
    priceEl.textContent = c2 ? "$1,499" : (c1 ? "$1,299" : "$1,099");
    priceEl.style.color = c2 ? "#D95D1E" : "#0E6EB8";

    let ripP = -1;
    if (t >= 1.05 && t <= 1.50) { ripP = (t - 1.05) / 0.45; rip.style.left = "570px"; rip.style.top = "565px"; }
    else if (t >= 2.35 && t <= 2.80) { ripP = (t - 2.35) / 0.45; rip.style.left = "580px"; rip.style.top = "677px"; }
    if (ripP >= 0) {
      rip.style.opacity = String(1 - ripP);
      rip.style.transform = `translate(-50%, -50%) scale(${0.3 + 1.4 * ripP})`;
    } else {
      rip.style.opacity = "0";
    }

    const hlPct = clamp((t - 1.5) / 1.6, 0, 1) * 100;
    hlEl.style.backgroundSize = `${hlPct.toFixed(1)}% 100%`;
  };
}

if (B.visual_type === "ui_configurator_crawl") {
  initUIConfigurator();
} else {
  initWebGL();
}

window.renderAt = function(t, dur, captionText) {
  updateHeroMetric(t);
  updateSceneFn(t, dur);
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
    sync3DLabels();
  }
  document.getElementById("cap-bar").textContent = captionText || "";
};

window.renderAt(0, 8, "");
window.__READY = true;
</script>
</body>
</html>
"""


SHORT_INTERACTIVE_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; -webkit-font-smoothing: antialiased; }
  html, body {
    width: 1080px; height: 1920px; overflow: hidden;
    font-family: "PingFang TC", "SF Pro Display", "Helvetica Neue", sans-serif;
    background: #101418;
  }

  /* Mode A: Dynamic Hook with Video/Pan Background + 5 Pop-in Die-Cut Stickers */
  #hook-wrap { position: absolute; inset: 0; display: none; overflow: hidden; }
  #hook-bg-img {
    position: absolute; inset: 0; width: 1080px; height: 1920px; object-fit: cover;
    transform-origin: center center;
  }
  #hook-bg-vid {
    position: absolute; inset: 0; width: 1080px; height: 1920px; object-fit: cover; display: none;
  }
  .hook-scrim {
    position: absolute; inset: 0;
    background: linear-gradient(180deg, rgba(10,14,18,0.52) 0%, rgba(10,14,18,0.16) 45%, rgba(10,14,18,0.72) 100%);
  }
  .pop-sticker {
    position: absolute; padding: 20px 40px; border-radius: 30px;
    font-size: 56px; font-weight: 800; letter-spacing: 0.01em;
    border: 6px solid #FFFFFF;
    box-shadow: 0 18px 40px rgba(0,0,0,0.34);
    white-space: nowrap; transform-origin: center center;
  }
  .st-white { background: #FFFFFF; color: #14161A; }
  .st-yellow { background: #FACC15; color: #161614; }
  .st-green { background: #16A34A; color: #FFFFFF; }

  .diecut-badge {
    position: absolute; width: 170px; height: 170px;
    filter: drop-shadow(0 16px 28px rgba(0,0,0,0.35));
    transform-origin: center center;
  }

  /* Mode B: Mint-Sage Interactive Step Phone Mockup (1:1 QrBoNy0wse8 Choreography) */
  #step-wrap {
    position: absolute; inset: 0; display: none;
    background: linear-gradient(165deg, #D9EED5 0%, #EDF7EA 48%, #DFF0DC 100%);
  }
  .top-pill {
    position: absolute; top: 86px; left: 62px;
    background: #101812; color: #FFF; padding: 12px 28px; border-radius: 999px;
    display: inline-flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 700;
  }
  .top-pill-dot { width: 14px; height: 14px; border-radius: 50%; background: #2ECC71; }
  .step-counter {
    position: absolute; top: 94px; right: 64px; font-size: 32px; font-weight: 800; color: #141A15;
  }
  .step-counter span { color: #8A9B8C; font-weight: 600; }
  .step-header {
    position: absolute; top: 162px; left: 62px; right: 62px;
    display: flex; align-items: baseline; gap: 22px;
  }
  .step-big-num { font-size: 122px; font-weight: 800; color: #17884C; line-height: 1; }
  .step-big-title { font-size: 78px; font-weight: 800; color: #141815; letter-spacing: -0.02em; }

  .flow-bar {
    position: absolute; top: 314px; left: 64px; right: 64px;
    display: flex; align-items: center; gap: 16px; font-size: 27px; font-weight: 700; color: #364438;
  }
  .flow-item.on { color: #17884C; background: rgba(23,136,76,0.12); padding: 4px 14px; border-radius: 999px; }

  .iphone-bezel {
    position: absolute; left: 165px; top: 386px; width: 750px; height: 1184px;
    background: #111613; border-radius: 74px; border: 6px solid #26784B;
    box-shadow: 0 28px 64px rgba(16, 38, 24, 0.22); padding: 20px;
  }
  .iphone-screen {
    position: relative; width: 100%; height: 100%; background: #F4F6F8;
    border-radius: 54px; overflow: hidden; display: flex; flex-direction: column;
  }
  .status-bar {
    height: 56px; display: flex; align-items: center; justify-content: space-between;
    padding: 10px 36px 0 36px; font-size: 20px; font-weight: 700; color: #181A1E;
  }
  .dyn-island {
    width: 150px; height: 34px; border-radius: 20px; background: #0B0D10;
  }
  .phone-banner {
    margin: 10px 22px 10px 22px; height: 198px; border-radius: 26px; overflow: hidden; position: relative;
    background: #1C222B;
  }
  .phone-banner img { width: 100%; height: 100%; object-fit: cover; }
  .phone-banner-tag {
    position: absolute; left: 18px; bottom: 12px;
    background: rgba(12,16,22,0.78); backdrop-filter: blur(8px);
    color: #FFF; font-size: 19px; font-weight: 700; padding: 5px 15px; border-radius: 999px;
  }

  .ios-list { padding: 4px 22px; display: flex; flex-direction: column; gap: 12px; flex: 1; }
  .ios-row {
    position: relative; background: #FFFFFF; border-radius: 24px; padding: 15px 22px;
    border: 3px solid #E6EBF0; display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 6px 16px rgba(0,0,0,0.03);
  }
  .ios-row.focused {
    border-color: #FACC15; background: #FFFEF7;
    box-shadow: 0 10px 26px rgba(250, 204, 21, 0.22);
  }
  .r-title { font-size: 27px; font-weight: 800; color: #16191D; }
  .r-sub { font-size: 20px; font-weight: 500; color: #68737D; margin-top: 4px; }
  .r-right { display: flex; align-items: center; gap: 14px; }
  .r-badge {
    font-size: 19px; font-weight: 700; padding: 5px 12px; border-radius: 10px;
    background: #E9F7EF; color: #17884C;
  }
  .ios-switch {
    width: 74px; height: 42px; border-radius: 999px; background: #CBD5E1; padding: 4px;
    display: flex; align-items: center;
  }
  .ios-switch.on { background: #2ECC71; justify-content: flex-end; }
  .ios-knob { width: 34px; height: 34px; border-radius: 50%; background: #FFF; box-shadow: 0 2px 6px rgba(0,0,0,0.22); }

  .phone-src {
    padding: 10px 30px 20px 30px; font-size: 19px; color: #737E88; font-weight: 600;
    text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }

  .side-callout {
    position: absolute; left: 28px; top: 760px; z-index: 12;
    background: #FACC15; color: #161612; font-size: 36px; font-weight: 900;
    padding: 16px 28px; border-radius: 24px; border: 6px solid #FFFFFF;
    box-shadow: 0 14px 30px rgba(0,0,0,0.18);
    transform-origin: center center;
  }

  .touch-dot {
    position: absolute; z-index: 15; width: 54px; height: 54px; border-radius: 50%;
    background: rgba(23, 136, 76, 0.32); border: 4px solid #FFFFFF;
    box-shadow: 0 6px 18px rgba(0,0,0,0.22); transform: translate(-50%, -50%);
  }
  .touch-rip {
    position: absolute; z-index: 14; width: 92px; height: 92px; border-radius: 50%;
    border: 4px solid #2ECC71; transform: translate(-50%, -50%) scale(0); opacity: 0;
  }

  .seg-bar {
    position: absolute; left: 62px; right: 62px; bottom: 54px; height: 14px;
    display: flex; gap: 12px; z-index: 15;
  }
  .seg-item { flex: 1; background: rgba(23, 136, 76, 0.20); border-radius: 999px; overflow: hidden; }
  .seg-fill { height: 100%; background: #17884C; border-radius: 999px; width: 0%; }

  .short-caption {
    position: absolute; left: 32px; right: 32px; top: 1628px; z-index: 25;
    text-align: center; font-size: 44px; font-weight: 800; line-height: 1.25;
    white-space: nowrap;
    color: #FFFFFF;
    -webkit-text-stroke: 6px #101318;
    paint-order: stroke fill;
    filter: drop-shadow(0 6px 14px rgba(0,0,0,0.36));
  }
</style>
</head>
<body>
<div id="hook-wrap">
  <img id="hook-bg-img" src="" alt="">
  <video id="hook-bg-vid" muted playsinline preload="auto"></video>
  <div class="hook-scrim"></div>
  <div class="top-pill" style="z-index:10"><span class="top-pill-dot"></span><span id="hk-pill"></span></div>
  <div id="hk-stickers"></div>
</div>

<div id="step-wrap">
  <div class="top-pill"><span class="top-pill-dot"></span><span id="st-pill"></span></div>
  <div class="step-counter" id="st-cnt"></div>
  <div class="step-header">
    <div class="step-big-num" id="st-num"></div>
    <div class="step-big-title" id="st-title"></div>
  </div>
  <div class="flow-bar" id="st-flow"></div>

  <div class="iphone-bezel">
    <div class="iphone-screen">
      <div class="status-bar"><span>9:41</span><div class="dyn-island"></div><span>5G 100%</span></div>
      <div class="phone-banner">
        <img id="ph-img" src="" alt="">
        <div class="phone-banner-tag" id="ph-tag"></div>
      </div>
      <div class="ios-list" id="ios-list"></div>
      <div class="phone-src" id="ph-src"></div>
    </div>
  </div>

  <div class="side-callout" id="side-tag"></div>
  <div class="touch-dot" id="t-dot"></div>
  <div class="touch-rip" id="t-rip"></div>
  <div class="seg-bar" id="seg-bar"></div>
</div>

<div class="short-caption" id="s-cap"></div>

<script>
window.BEAT = __BEAT_JSON__;
window.__READY = false;

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function easeOutBack(x) {
  x = clamp(x, 0, 1);
  const c1 = 1.75, c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
}
function easeInOutCubic(x) { x = clamp(x, 0, 1); return x < 0.5 ? 4*x*x*x : 1 - Math.pow(-2*x + 2, 3)/2; }

const B = window.BEAT;
let updateShortFn = async function(t, dur) {};

async function setup() {
  if (B.visual_type === "short_hook_dynamic") {
    const hw = document.getElementById("hook-wrap");
    hw.style.display = "block";
    document.getElementById("hk-pill").textContent = (B.topic_pill || "Hey Digital 實測避坑").slice(0, 14);

    const imgEl = document.getElementById("hook-bg-img");
    const vidEl = document.getElementById("hook-bg-vid");
    let useVideo = false;
    if (B.bg_video_url) {
      vidEl.src = B.bg_video_url;
      vidEl.style.display = "block";
      imgEl.style.display = "none";
      await new Promise(res => {
        vidEl.onloadeddata = () => { useVideo = true; res(); };
        vidEl.onerror = () => res();
        setTimeout(res, 1500);
      });
    }
    if (!useVideo && B.bg_img_url) {
      imgEl.src = B.bg_img_url;
      imgEl.style.display = "block";
      await new Promise(res => { imgEl.onload = res; imgEl.onerror = res; setTimeout(res, 1000); });
    }

    const stickers = (B.pop_stickers && B.pop_stickers.length) ? B.pop_stickers.slice(0, 5) : [
      { text: "帳面便宜卻踩雷？", style: "white" },
      { text: "升級完一樣貴！", style: "yellow" },
      { text: "別只看跑分數據", style: "white" },
      { text: "5 個隱藏刀法", style: "green" },
      { text: "看完再下單 →", style: "yellow" }
    ];
    const layout = [
      { x: 95,  y: 215, rot: -3.2 },
      { x: 340, y: 405, rot: 3.4 },
      { x: 88,  y: 620, rot: -2.4 },
      { x: 330, y: 835, rot: 2.8 },
      { x: 165, y: 1065, rot: -2.0 }
    ];
    const stBox = document.getElementById("hk-stickers");
    const stEls = stickers.map((s, i) => {
      const el = document.createElement("div");
      const cls = s.style === "yellow" ? "st-yellow" : (s.style === "green" ? "st-green" : "st-white");
      el.className = `pop-sticker ${cls}`;
      const cleanTxt = (s.text || "").slice(0, 18);
      el.textContent = cleanTxt;
      if (cleanTxt.length > 10) el.style.fontSize = "48px";
      el.style.left = `${layout[i].x}px`;
      el.style.top = `${layout[i].y}px`;
      stBox.appendChild(el);
      return { el, rot: layout[i].rot, idx: i };
    });

    // Die-cut vector magnifier icon
    const icon = document.createElement("div");
    icon.className = "diecut-badge";
    icon.style.left = "760px"; icon.style.top = "565px";
    icon.innerHTML = `<svg viewBox="0 0 160 160"><circle cx="70" cy="70" r="48" fill="#67CCF8" stroke="#FFF" stroke-width="12"/><circle cx="70" cy="70" r="42" fill="none" stroke="#181A1F" stroke-width="5"/><line x1="106" y1="106" x2="144" y2="144" stroke="#FFF" stroke-width="24" stroke-linecap="round"/><line x1="106" y1="106" x2="144" y2="144" stroke="#F06428" stroke-width="14" stroke-linecap="round"/></svg>`;
    stBox.appendChild(icon);

    updateShortFn = async (t, dur) => {
      if (useVideo && vidEl.duration > 0.2) {
        const targetT = t % vidEl.duration;
        if (Math.abs(vidEl.currentTime - targetT) > 0.03) {
          await Promise.race([
            new Promise(r => { vidEl.onseeked = r; vidEl.currentTime = targetT; }),
            new Promise(r => setTimeout(r, 70))
          ]);
        }
      } else {
        const sc = 1.0 + 0.08 * (t / Math.max(3, dur));
        imgEl.style.transform = `scale(${sc.toFixed(4)})`;
      }
      const stepSpan = (Math.max(4, dur) * 0.66) / stEls.length;
      stEls.forEach(({ el, rot, idx }) => {
        const t0 = 0.12 + idx * stepSpan;
        const p = easeOutBack((t - t0) / 0.36);
        const op = clamp((t - t0) / 0.14, 0, 1);
        el.style.opacity = String(op);
        el.style.transform = `rotate(${rot}deg) scale(${Math.max(0.01, p)})`;
      });
      const ip = easeOutBack((t - 0.55) / 0.42);
      icon.style.opacity = String(clamp((t - 0.55) / 0.15, 0, 1));
      icon.style.transform = `rotate(${8 + Math.sin(t*3)*4}deg) scale(${Math.max(0.01, ip)})`;
    };
  }
  else {
    // Mint-Sage Interactive Step Phone Mockup
    const sw = document.getElementById("step-wrap");
    sw.style.display = "block";
    const sIdx = B.step_idx || 1;
    const sTot = B.step_total || 5;
    document.getElementById("st-pill").textContent = (B.topic_pill || "Hey Digital 實測避坑").slice(0, 14);
    document.getElementById("st-cnt").innerHTML = `${String(sIdx).padStart(2,"0")} <span>/ ${String(sTot).padStart(2,"0")}</span>`;
    document.getElementById("st-num").textContent = String(sIdx).padStart(2,"0");
    document.getElementById("st-title").textContent = (B.step_title || "關鍵規格差異").slice(0, 8);

    const flows = (B.flow_steps && B.flow_steps.length) ? B.flow_steps.slice(0, 3) : ["看規格表", "實測驗證", "選購結論"];
    const circles = ["①", "②", "③"];
    const flowBar = document.getElementById("st-flow");
    const flowEls = [];
    flows.forEach((f, i) => {
      const sp = document.createElement("span");
      sp.className = "flow-item";
      sp.textContent = `${circles[i]} ${String(f).slice(0, 7)}`;
      flowBar.appendChild(sp);
      flowEls.push(sp);
      if (i < flows.length - 1) {
        const ar = document.createElement("span");
        ar.style.color = "#96AD99"; ar.textContent = "→";
        flowBar.appendChild(ar);
      }
    });

    const phImg = document.getElementById("ph-img");
    if (B.bg_img_url) {
      phImg.src = B.bg_img_url;
      await new Promise(res => { phImg.onload = res; phImg.onerror = res; setTimeout(res, 1000); });
    }
    const src = B.source_detail || {};
    document.getElementById("ph-tag").textContent = `${src.publisher || "公開實測數據"} · ${B.source_id || "SRC-01"}`;
    document.getElementById("ph-src").textContent = `公開驗證來源：${src.publisher || "Tech Spec"} — ${(src.title || "").slice(0, 26)}`;

    const rows = (B.ui_rows && B.ui_rows.length) ? B.ui_rows.slice(0, 4) : [
      { title: "標準起步規格", subtitle: "輕量文書與串流影音夠用", badge: "省預算", toggle_on: true },
      { title: "升級記憶體與硬碟", subtitle: "加價幅度迅速貼近高階款", badge: "注意價差", toggle_on: false },
      { title: "高負載散熱與頻寬", subtitle: "長時間輸出維持穩定幀率", badge: "實測勝出", toggle_on: true },
      { title: "Hey Digital 選購結論", subtitle: "按主力工作流精準分流", badge: "推薦必看", toggle_on: true }
    ];
    const listEl = document.getElementById("ios-list");
    const rowDom = rows.map((r, ri) => {
      const d = document.createElement("div");
      d.className = "ios-row";
      d.innerHTML = `
        <div style="max-width:440px">
          <div class="r-title">${String(r.title || "").slice(0, 15)}</div>
          <div class="r-sub">${String(r.subtitle || "").slice(0, 22)}</div>
        </div>
        <div class="r-right">
          <span class="r-badge">${String(r.badge || "重點").slice(0, 6)}</span>
          <div class="ios-switch" id="sw-${ri}"><div class="ios-knob"></div></div>
        </div>
      `;
      listEl.appendChild(d);
      return { el: d, sw: d.querySelector(".ios-switch"), wantOn: r.toggle_on !== false };
    });

    const sideTag = document.getElementById("side-tag");
    sideTag.textContent = (B.callout_tag || "必看刀法").slice(0, 6);

    const segBar = document.getElementById("seg-bar");
    const segFills = [];
    for (let i = 0; i < sTot; i++) {
      const it = document.createElement("div"); it.className = "seg-item";
      const fl = document.createElement("div"); fl.className = "seg-fill";
      it.appendChild(fl); segBar.appendChild(it);
      segFills.push(fl);
    }

    const tDot = document.getElementById("t-dot");
    const tRip = document.getElementById("t-rip");

    updateShortFn = async (t, dur) => {
      const prog = clamp(t / Math.max(3, dur), 0, 1);
      const activeFlow = Math.min(flowEls.length - 1, Math.floor(prog * flowEls.length));
      flowEls.forEach((el, i) => { el.className = "flow-item" + (i === activeFlow ? " on" : ""); });

      const activeRow = Math.min(rowDom.length - 1, Math.floor(prog * rowDom.length));
      rowDom.forEach((rd, ri) => {
        rd.el.className = "ios-row" + (ri === activeRow ? " focused" : "");
        const switched = (t >= (ri + 0.35) * (dur / rowDom.length)) && rd.wantOn;
        rd.sw.className = "ios-switch" + (switched ? " on" : "");
      });

      // Touch dot glides smoothly down the exact live DOM centers of the iOS switches!
      const swRects = rowDom.map(rd => rd.sw.getBoundingClientRect());
      const rowCentersY = swRects.map((rc, i) => rc.height > 0 ? (rc.top + rc.height * 0.5) : (740 + i * 138));
      const swCenterX = (swRects[0] && swRects[0].width > 0) ? (swRects[0].left + swRects[0].width * 0.5) : 812;
      const curFloat = clamp(prog * (rowDom.length - 0.15), 0, rowDom.length - 1);
      const idx0 = Math.floor(curFloat);
      const idx1 = Math.min(rowDom.length - 1, idx0 + 1);
      const frac = easeInOutCubic(curFloat - idx0);
      const ty = rowCentersY[idx0] + (rowCentersY[idx1] - rowCentersY[idx0]) * frac;
      tDot.style.left = `${swCenterX}px`; tDot.style.top = `${ty}px`;

      const slotDur = dur / rowDom.length;
      const localSlot = (t % slotDur) / slotDur;
      if (localSlot >= 0.32 && localSlot <= 0.68) {
        const rp = (localSlot - 0.32) / 0.36;
        tRip.style.left = `${swCenterX}px`; tRip.style.top = `${rowCentersY[activeRow]}px`;
        tRip.style.opacity = String(1 - rp);
        tRip.style.transform = `translate(-50%, -50%) scale(${0.4 + 1.2 * rp})`;
      } else {
        tRip.style.opacity = "0";
      }

      const sp = easeOutBack((t - 0.45) / 0.42);
      sideTag.style.opacity = String(clamp((t - 0.45) / 0.18, 0, 1));
      sideTag.style.transform = `rotate(-5deg) scale(${Math.max(0.01, sp)})`;

      segFills.forEach((fl, si) => {
        if (si < sIdx - 1) fl.style.width = "100%";
        else if (si === sIdx - 1) fl.style.width = `${(prog * 100).toFixed(1)}%`;
        else fl.style.width = "0%";
      });
    };
  }
  await updateShortFn(0, 6);
  window.__READY = true;
}

window.renderAt = async function(t, dur, captionText) {
  await updateShortFn(t, dur);
  document.getElementById("s-cap").textContent = captionText || "";
};

setup();
</script>
</body>
</html>
"""


class MotionStudioRenderer:
    """
    Persistent Metal WebGL2 + HTML5/CSS3 Frame Renderer using local Chrome + Playwright.
    Streams deterministic 30fps JPEG frames straight into FFmpeg `libx264` with zero temp disk bloat.
    """

    def __init__(self, repo_root: Path, run_dir: Path):
        self.repo_root = repo_root
        self.run_dir = run_dir
        self._httpd: Optional[_ThreadedServer] = None
        self._pw = None
        self._browser = None
        self._page_long = None
        self._page_short = None
        self.base_url = ""

    def __enter__(self) -> "MotionStudioRenderer":
        _QuietStaticHandler.root_map = {
            "/repo": self.repo_root,
            "/run": self.run_dir,
        }
        _QuietStaticHandler.dynamic_html = {}
        self._httpd = _ThreadedServer(("127.0.0.1", 0), _QuietStaticHandler)
        port = self._httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{port}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        launch_kwargs = {
            "headless": True,
            "args": [
                "--use-gl=angle",
                "--use-angle=metal",
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--autoplay-policy=no-user-gesture-required",
                "--hide-scrollbars",
            ],
        }
        if os.path.exists(CHROME_EXEC):
            launch_kwargs["executable_path"] = CHROME_EXEC
        self._browser = self._pw.chromium.launch(**launch_kwargs)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        finally:
            if self._httpd:
                self._httpd.shutdown()

    def _get_long_page(self):
        if self._page_long is None:
            self._page_long = self._browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        return self._page_long

    def _get_short_page(self):
        if self._page_short is None:
            self._page_short = self._browser.new_page(viewport={"width": 1080, "height": 1920}, device_scale_factor=1)
        return self._page_short

    def render_long_webgl_beat(
        self,
        beat: dict,
        wav_path: Path,
        caption_timings: list[dict],
        real_dur: float,
        out_mp4: Path,
        fps: int = 30,
    ) -> list[dict]:
        """Render a `1920x1080@30fps` Three.js WebGL2 or Interactive Configurator beat to `out_mp4` and return relative SFX events."""
        page = self._get_long_page()
        html_str = (
            COMMON_LONG_HTML_TEMPLATE
            .replace("__THREE_URL__", f"{self.base_url}/repo/assets/vendor/three.min.js")
            .replace("__BEAT_JSON__", json.dumps(beat, ensure_ascii=False))
        )
        _QuietStaticHandler.dynamic_html["/long_scene.html"] = html_str.encode("utf-8")
        page.goto(f"{self.base_url}/long_scene.html", wait_until="load")
        page.wait_for_function("window.__READY === true", timeout=5000)

        n_frames = max(fps, int(round(real_dur * fps)))
        ff_cmd = [
            "ffmpeg", "-y",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(fps), "-i", "-",
            "-i", str(wav_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-t", f"{real_dur:.3f}",
            str(out_mp4),
        ]
        proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None

        for fi in range(n_frames):
            t_loc = fi / fps
            cap = caption_for_timestamp(beat, caption_timings, t_loc, real_dur)
            page.evaluate("([t, d, c]) => window.renderAt(t, d, c)", [t_loc, real_dur, cap])
            jpg = page.screenshot(type="jpeg", quality=94)
            proc.stdin.write(jpg)

        proc.stdin.close()
        proc.wait()

        # Return beat-relative SFX cues matching exact visual animation moments!
        vtype = beat.get("visual_type")
        sfx_rel = []
        if vtype == "ui_configurator_crawl":
            sfx_rel.extend([{"offset": 1.05, "kind": "click", "pan": -0.2}, {"offset": 2.35, "kind": "click", "pan": -0.2}])
        elif vtype == "webgl_stacked_tco":
            sfx_rel.extend([{"offset": 1.18, "kind": "pop", "pan": 0.05}, {"offset": 1.73, "kind": "pop", "pan": 0.05}])
        elif vtype == "webgl_memory_lanes" and beat.get("secondary_metric"):
            sfx_rel.append({"offset": 1.25, "kind": "pop", "pan": -0.25})
        return [e for e in sfx_rel if e["offset"] < real_dur - 0.1]

    def render_short_interactive_beat(
        self,
        beat: dict,
        wav_path: Path,
        caption_timings: list[dict],
        real_dur: float,
        out_mp4: Path,
        step_idx: int,
        step_total: int,
        bg_img_rel_run: Optional[str] = None,
        bg_vid_rel_run: Optional[str] = None,
        fps: int = 30,
    ) -> list[dict]:
        """Render a `1080x1920@30fps` Interactive Short beat (`short_hook_dynamic` or `short_phone_interactive`)."""
        page = self._get_short_page()
        beat_payload = dict(beat)
        beat_payload["step_idx"] = step_idx
        beat_payload["step_total"] = step_total
        if bg_img_rel_run:
            beat_payload["bg_img_url"] = f"{self.base_url}/run/{bg_img_rel_run.lstrip('/')}"
        if bg_vid_rel_run:
            beat_payload["bg_video_url"] = f"{self.base_url}/run/{bg_vid_rel_run.lstrip('/')}"

        html_str = SHORT_INTERACTIVE_HTML_TEMPLATE.replace("__BEAT_JSON__", json.dumps(beat_payload, ensure_ascii=False))
        _QuietStaticHandler.dynamic_html["/short_scene.html"] = html_str.encode("utf-8")
        page.goto(f"{self.base_url}/short_scene.html", wait_until="load")
        page.wait_for_function("window.__READY === true", timeout=5000)

        n_frames = max(fps, int(round(real_dur * fps)))
        ff_cmd = [
            "ffmpeg", "-y",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-r", str(fps), "-i", "-",
            "-i", str(wav_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-t", f"{real_dur:.3f}",
            str(out_mp4),
        ]
        proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None

        for fi in range(n_frames):
            t_loc = fi / fps
            cap = caption_for_timestamp(beat, caption_timings, t_loc, real_dur)
            page.evaluate("([t, d, c]) => window.renderAt(t, d, c)", [t_loc, real_dur, cap])
            jpg = page.screenshot(type="jpeg", quality=94)
            proc.stdin.write(jpg)

        proc.stdin.close()
        proc.wait()

        sfx_rel = []
        if beat.get("visual_type") == "short_hook_dynamic":
            n_st = min(5, len(beat.get("pop_stickers") or [1, 2, 3, 4, 5]))
            span = (max(4.0, real_dur) * 0.66) / n_st
            for i in range(n_st):
                sfx_rel.append({"offset": 0.12 + i * span, "kind": "pop", "pan": -0.15 + 0.08 * i})
        else:
            sfx_rel.append({"offset": 0.45, "kind": "pop", "pan": -0.25})
            n_rows = min(4, len(beat.get("ui_rows") or [1, 2, 3, 4]))
            slot = real_dur / n_rows
            for ri in range(n_rows):
                sfx_rel.append({"offset": (ri + 0.35) * slot, "kind": "click", "pan": 0.2})
        return [e for e in sfx_rel if e["offset"] < real_dur - 0.1]
