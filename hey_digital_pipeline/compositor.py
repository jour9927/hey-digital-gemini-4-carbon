#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon V2 — Flagship Video, Interactive Shorts & Spotlight Thumbnail Compositor
====================================================================================================
V2 Complete Station-by-Station Upgrade matched to `@hey_digitall`:
1. Long-Form (`1920x1080 @ 30fps`):
   - `chapter_dark_card`: 1.85s musical transition card (animated slide-up & glow on matte micro-grid,
     giant outlined chapter numeral, gold eyebrow, airy `whoosh` SFX + un-ducked Rhodes chord bloom).
   - True 3D `Three.js` WebGL2 + Interactive UI scenes via `MotionStudioRenderer` (`webgl_memory_lanes`,
     `webgl_stacked_tco`, `webgl_ssd_particles`, `webgl_chip_die`, `webgl_dual_laptop`, `ui_configurator_crawl`).
   - Genuine 30fps Video B-Roll (`cinema_broll_video`): Streams harvested `yt-dlp` 1080p hardware B-roll
     clips and RTX Pro 6000 `LTX-Video` I2V clips (with high-res parallax fallback), overlaid with crisp
     chapter HUD, verifiable source badge, floating crawled news card, and Whisper-aligned captions.
   - End-of-video `Public Crawled Sources Slate` (`3.8s`).
   - Mastered with 86 BPM Fender Rhodes Electric-Piano Chill Groove + Sidechain Ducking + Tactile SFX (`-14.2 LUFS`).
2. Traffic-Funnel Short (`1080x1920 @ 30fps`):
   - `short_hook_dynamic`: Live vertical video B-roll + 5 sequential spring-bouncing stickers & vector icon + `pop` SFX.
   - `short_phone_interactive`: Mint-sage layout + full-height emerald iPhone Pro bezel + 4 interactive iOS rows
     with live touch-cursor ripples, flipping green switches, yellow focus highlight, and `click` SFX.
   - `short_broll_summary` & `cta_funnel_endcard`: Vertical motion B-roll recap + animated Long-Form thumbnail funnel CTA.
"""
from __future__ import annotations

import functools
import glob
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from hey_digital_pipeline.audio_studio import (
    build_sfx_track,
    generate_rhodes_chill_bgm,
    master_video_with_sidechain,
)
from hey_digital_pipeline.motion_studio import MotionStudioRenderer, caption_for_timestamp


def _discover_pingfang_ttc() -> list[str]:
    direct = ["/System/Library/Fonts/PingFang.ttc"]
    assets = glob.glob("/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/PingFang.ttc")
    return [p for p in (direct + assets) if os.path.exists(p)]


CJK_FONT_CANDIDATES_BOLD = _discover_pingfang_ttc() + [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]

LATIN_THIN_CANDIDATES = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


@functools.lru_cache(maxsize=64)
def get_cjk_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    for fp in CJK_FONT_CANDIDATES_BOLD:
        if os.path.exists(fp):
            try:
                if fp.endswith("PingFang.ttc"):
                    idx = 10 if bold else 2
                elif fp.endswith("Hiragino Sans GB.ttc"):
                    idx = 2 if bold else 0
                else:
                    idx = 0
                return ImageFont.truetype(fp, size=size, index=idx)
            except Exception:
                try:
                    return ImageFont.truetype(fp, size=size)
                except Exception:
                    continue
    return ImageFont.load_default()


@functools.lru_cache(maxsize=64)
def get_latin_thin_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in LATIN_THIN_CANDIDATES:
        if os.path.exists(fp):
            try:
                idx = 7 if fp.endswith("HelveticaNeue.ttc") else 0
                return ImageFont.truetype(fp, size=size, index=idx)
            except Exception:
                try:
                    return ImageFont.truetype(fp, size=size)
                except Exception:
                    continue
    return get_cjk_font(size, bold=False)


def cover_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    img = img.convert("RGB")
    iw, ih = img.size
    scale = max(target_w / iw, target_h / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - target_w) // 2
    top = (nh - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def draw_adaptive_caption(canvas: Image.Image, text: str, is_light_bg: bool, is_vertical: bool = False):
    """
    Render strictly SINGLE-LINE `@hey_digitall` caption (NEVER wraps to 2 lines or chops mid-word;
    dynamically scales font down slightly only if an unusually dense line approaches screen edges).
    """
    if not text:
        return
    w, h = canvas.size
    font_size = 44 if is_vertical else 40
    max_px = int(w * (0.90 if is_vertical else 0.88))
    draw = ImageDraw.Draw(canvas, "RGBA")

    font = get_cjk_font(font_size, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    while tw > max_px and font_size > 28:
        font_size -= 2
        font = get_cjk_font(font_size, bold=True)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]

    x = (w - tw) // 2
    y = int(h * (0.848 if is_vertical else 0.908))
    if is_light_bg and not is_vertical:
        draw.text((x, y + 1), text, font=font, fill=(255, 255, 255, 135))
        draw.text((x, y), text, font=font, fill=(22, 22, 24, 255))
    else:
        draw.text((x + 2, y + 4), text, font=font, fill=(0, 0, 0, 195))
        draw.text(
            (x, y),
            text,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=4 if is_vertical else 2,
            stroke_fill=(10, 12, 16, 245),
        )


def draw_source_badge(draw: ImageDraw.ImageDraw, badge_text: str, x: int, y: int, light_theme: bool):
    if not badge_text:
        return
    font = get_cjk_font(21, bold=False)
    bb = draw.textbbox((0, 0), badge_text, font=font)
    pw = (bb[2] - bb[0]) + 38
    ph = 38
    if light_theme:
        draw.rounded_rectangle([x, y, x + pw, y + ph], radius=19, fill=(255, 255, 255, 205), outline=(206, 202, 192, 255), width=1)
        draw.ellipse([x + 14, y + 14, x + 24, y + 24], fill=(14, 124, 71, 255))
        draw.text((x + 32, y + 6), badge_text, font=font, fill=(68, 67, 63, 255))
    else:
        draw.rounded_rectangle([x, y, x + pw, y + ph], radius=19, fill=(14, 16, 22, 195), outline=(255, 255, 255, 65), width=1)
        draw.ellipse([x + 14, y + 14, x + 24, y + 24], fill=(229, 153, 52, 255))
        draw.text((x + 32, y + 6), badge_text, font=font, fill=(232, 236, 242, 245))


def render_chapter_dark_card_animated(beat: dict, bg_img: Optional[Image.Image], progress: float) -> Image.Image:
    """Render animated 1.85s musical chapter transition card (smooth cubic-out slide & line wipe)."""
    w, h = 1920, 1080
    canvas = Image.new("RGB", (w, h), (11, 12, 15))
    draw = ImageDraw.Draw(canvas, "RGBA")

    for gx in range(40, w, 36):
        for gy in range(40, h, 36):
            draw.point((gx, gy), fill=(36, 39, 46, 145))

    ease = 1.0 - (1.0 - min(1.0, progress / 0.55)) ** 3 if progress < 0.55 else 1.0

    if bg_img is not None:
        sub = cover_resize(bg_img, 980, 620)
        sub = ImageEnhance.Brightness(sub).enhance(0.35)
        mask = Image.new("L", (980, 620), 0)
        ImageDraw.Draw(mask).rounded_rectangle([40, 40, 960, 600], radius=36, fill=int(215 * ease))
        mask = mask.filter(ImageFilter.GaussianBlur(42))
        canvas.paste(sub, (int(960 - 40 * ease), 440), mask)

    draw = ImageDraw.Draw(canvas, "RGBA")
    ch_num = str(beat.get("chapter_num") or "01").zfill(2)
    ch_en = (beat.get("chapter_en") or "DEEP DIVE").upper()
    ch_title = beat.get("chapter_title") or "核心拆解"
    subtitle = beat.get("chapter_sub") or (beat.get("narration") or "")[:24].strip("，。！？ ")

    y_shift = int(28 * (1.0 - ease))
    alpha = int(255 * ease)

    f_giant = get_latin_thin_font(470)
    draw.text((48, 235 + y_shift), ch_num, font=f_giant, fill=(11, 12, 15, 0), stroke_width=2, stroke_fill=(135, 140, 152, int(215 * ease)))

    rx = 985
    f_en = get_cjk_font(23, bold=True)
    spaced_en = " ".join(list(f"CHAPTER {ch_num} / {ch_en}"))
    draw.text((rx, 345 + y_shift), spaced_en[:56], font=f_en, fill=(229, 153, 52, alpha))

    line_end = rx + int(785 * ease)
    if line_end > rx + 4:
        draw.line([(rx, 398), (line_end, 398)], fill=(76, 80, 90, alpha), width=2)

    f_title = get_cjk_font(68, bold=True)
    draw.text((rx, 432 + y_shift), ch_title, font=f_title, fill=(255, 255, 255, alpha))

    if subtitle:
        f_sub = get_cjk_font(30, bold=False)
        draw.text((rx, 532 + y_shift), subtitle, font=f_sub, fill=(148, 154, 165, alpha))
    return canvas


def overlay_crawled_news_card(canvas: Image.Image, beat: dict):
    src = beat.get("source_detail")
    if not src:
        return
    draw = ImageDraw.Draw(canvas, "RGBA")
    cx, cy, cw, ch = 96, 560, 660, 250
    region = canvas.crop((cx, cy, cx + cw, cy + ch)).filter(ImageFilter.GaussianBlur(14))
    region = ImageEnhance.Brightness(region).enhance(0.42)
    mask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, cw, ch], radius=24, fill=235)
    canvas.paste(region, (cx, cy), mask)

    draw.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=24, outline=(255, 255, 255, 80), width=2)
    draw.ellipse([cx + 22, cy + 20, cx + 34, cy + 32], fill=(255, 95, 86, 230))
    draw.ellipse([cx + 42, cy + 20, cx + 54, cy + 32], fill=(255, 189, 46, 230))
    draw.ellipse([cx + 62, cy + 20, cx + 74, cy + 32], fill=(39, 201, 63, 230))

    f_dom = get_cjk_font(19, bold=True)
    domain_str = f"{src.get('publisher', 'Public Web')}  ·  {src.get('domain', '')}  ({src.get('published', '')[:10]})"
    draw.text((cx + 92, cy + 14), domain_str[:46], font=f_dom, fill=(190, 200, 215, 245))
    draw.line([(cx + 20, cy + 48), (cx + cw - 20, cy + 48)], fill=(255, 255, 255, 42), width=1)

    f_t = get_cjk_font(25, bold=True)
    title = src.get("title", "")
    draw.text((cx + 24, cy + 64), title[:25], font=f_t, fill=(255, 255, 255, 255))
    if len(title) > 25:
        draw.text((cx + 24, cy + 98), title[25:50] + ("…" if len(title) > 50 else ""), font=f_t, fill=(255, 255, 255, 255))

    metrics = src.get("extracted_metrics") or []
    f_m = get_cjk_font(20, bold=False)
    snip = ("實測數據：" + " / ".join(metrics[:4])) if metrics else (src.get("summary", "")[:50] + "…")
    draw.text((cx + 24, cy + 148), snip[:36], font=f_m, fill=(240, 202, 134, 255))
    draw.text((cx + 24, cy + 194), f"公開驗證編號 {src.get('source_id', '')} · 完整連結附於資訊欄", font=get_cjk_font(18, False), fill=(155, 165, 180, 230))


def render_sources_slate(cited_sources: list[dict]) -> Image.Image:
    w, h = 1920, 1080
    canvas = Image.new("RGB", (w, h), (13, 15, 19))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((110, 78), "PUBLIC DATA & CRAWLED SOURCES", font=get_cjk_font(22, True), fill=(229, 153, 52, 255))
    draw.text((110, 114), "本期影片公開爬蟲資訊與實測參考來源", font=get_cjk_font(46, True), fill=(255, 255, 255, 255))
    draw.line([(110, 186), (1810, 186)], fill=(55, 60, 70, 255), width=2)

    f_id = get_cjk_font(21, True)
    f_tt = get_cjk_font(24, True)
    f_url = get_cjk_font(18, False)
    for idx, s in enumerate(cited_sources[:6]):
        y = 216 + idx * 124
        draw.rounded_rectangle([110, y, 1810, y + 106], radius=16, fill=(22, 25, 32, 255), outline=(45, 50, 62, 255), width=1)
        draw.rounded_rectangle([132, y + 20, 240, y + 56], radius=10, fill=(229, 153, 52, 42), outline=(229, 153, 52, 190), width=1)
        draw.text((148, y + 24), s.get("source_id", f"SRC-{idx+1:02d}"), font=f_id, fill=(240, 202, 134, 255))
        draw.text((262, y + 18), f"{s.get('publisher', '')} — {s.get('title', '')[:56]}", font=f_tt, fill=(242, 245, 250, 255))
        draw.text((262, y + 60), s.get("url", "")[:98], font=f_url, fill=(138, 148, 165, 255))
    return canvas


def render_thumbnail(plan: dict, hero_img: Optional[Image.Image], out_path: Path) -> Path:
    w, h = 1920, 1080
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - w * 0.5) / (w * 0.62)) ** 2 + ((yy - h * 0.68) / (h * 0.58)) ** 2)
    dist = np.clip(dist, 0.0, 1.0)
    c_r = (34 * (1 - dist) + 8 * dist).astype(np.uint8)
    c_g = (38 * (1 - dist) + 9 * dist).astype(np.uint8)
    c_b = (46 * (1 - dist) + 12 * dist).astype(np.uint8)
    canvas = Image.fromarray(np.dstack([c_r, c_g, c_b]), mode="RGB")

    if hero_img is not None:
        prod = cover_resize(hero_img, 1340, 670)
        mask = Image.new("L", (1340, 670), 0)
        ImageDraw.Draw(mask).rounded_rectangle([30, 20, 1310, 650], radius=48, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(24))
        canvas.paste(prod, ((w - 1340) // 2, 410), mask)

    grad = Image.new("RGBA", (w, 470), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad, "RGBA")
    for gy in range(470):
        alpha = int(218 * max(0.0, 1.0 - (gy / 460.0) ** 1.4))
        gd.line([(0, gy), (w, gy)], fill=(8, 9, 12, alpha))
    canvas.paste(grad, (0, 0), grad)

    th = plan.get("thumbnail") or {}
    line1 = th.get("line1") or plan.get("title_long", "")[:10]
    line2 = th.get("line2") or plan.get("title_long", "")[10:20]
    gold = th.get("gold_phrase") or ""

    draw = ImageDraw.Draw(canvas, "RGBA")
    for li, ln in enumerate([line1, line2]):
        if not ln:
            continue
        f_big = get_cjk_font(126 if len(ln) <= 11 else 104, bold=True)
        bb = draw.textbbox((0, 0), ln, font=f_big)
        tw = bb[2] - bb[0]
        x = (w - tw) // 2
        y = 62 + li * 156
        draw.text((x + 4, y + 7), ln, font=f_big, fill=(0, 0, 0, 220), stroke_width=3, stroke_fill=(0, 0, 0, 220))
        if gold and gold in ln:
            pre, post = ln.split(gold, 1)
            cx = x
            if pre:
                draw.text((cx, y), pre, font=f_big, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))
                cx = draw.textbbox((cx, y), pre, font=f_big)[2]
            draw.text((cx, y), gold, font=f_big, fill=(240, 202, 134, 255), stroke_width=2, stroke_fill=(240, 202, 134, 255))
            cx = draw.textbbox((cx, y), gold, font=f_big)[2]
            if post:
                draw.text((cx, y), post, font=f_big, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))
        else:
            draw.text((x, y), ln, font=f_big, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), quality=95)
    return out_path


def _compute_subshot_windows(matched_subshots: list[dict], caption_timings: list[dict], real_dur: float) -> list[tuple[float, float, dict]]:
    """Align B-roll sub-shot cut points directly to Whisper forced-aligned clause boundaries!"""
    if not matched_subshots:
        return []
    k = len(matched_subshots)
    if k == 1:
        return [(0.0, real_dur, matched_subshots[0])]

    cuts = [0.0]
    if caption_timings and len(caption_timings) >= k:
        for i in range(1, k):
            cap_idx = max(0, min(len(caption_timings) - 2, round(i * len(caption_timings) / k) - 1))
            t_cut = float(caption_timings[cap_idx]["end"])
            # Keep every sub-shot comfortably watchable (at least 1.8s)
            t_cut = max(cuts[-1] + 1.8, min(real_dur - (k - i) * 1.8, t_cut))
            cuts.append(t_cut)
    else:
        for i in range(1, k):
            cuts.append(real_dur * (i / k))
    cuts.append(real_dur)
    return [(cuts[i], cuts[i + 1], matched_subshots[i]) for i in range(k)]


def _render_broll_video_beat(
    beat: dict,
    wav_path: Path,
    video_src: Optional[Path],
    fallback_img: Optional[Image.Image],
    caption_timings: list[dict],
    real_dur: float,
    out_mp4: Path,
    fps: int = 30,
    is_vertical: bool = False,
    show_news_card: bool = False,
):
    """
    V3 Phrase-Synced Multi-Shot Third-Party B-Roll Renderer:
    Cuts frame-accurately across `beat["matched_subshots"]` right on the Whisper spoken clause
    timestamps, updating the on-screen source badge for each real hardware shot!
    """
    tw, th = (1080, 1920) if is_vertical else (1920, 1080)
    n_frames = max(fps, int(round(real_dur * fps)))

    ff_enc = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{tw}x{th}", "-r", str(fps), "-i", "-",
        "-i", str(wav_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-t", f"{real_dur:.3f}",
        str(out_mp4),
    ]
    enc_proc = subprocess.Popen(ff_enc, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert enc_proc.stdin is not None

    frame_bytes = tw * th * 3
    vf = f"scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},fps={fps}"

    matched_subs = [s for s in (beat.get("matched_subshots") or []) if s.get("mp4") and Path(s["mp4"]).exists()]
    windows = _compute_subshot_windows(matched_subs, caption_timings, real_dur) if (matched_subs and not is_vertical) else []

    active_idx = -1
    dec_proc: Optional[subprocess.Popen] = None
    active_badge = beat.get("source_badge") or ""
    sid = beat.get("source_id") or ""
    if active_badge and sid and sid not in active_badge:
        active_badge = f"{active_badge} · {sid}"

    def _open_decoder(mp4_p: Path) -> subprocess.Popen:
        return subprocess.Popen(
            ["ffmpeg", "-v", "error", "-stream_loop", "-1", "-i", str(mp4_p), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    if not windows and video_src is not None and video_src.exists():
        dec_proc = _open_decoder(video_src)

    ch_num = str(beat.get("chapter_num") or "01").zfill(2)
    hud_txt = f"{ch_num}  ·  {beat.get('chapter_title', '實測畫面')}"
    f_hud = get_cjk_font(23, bold=False)

    hi_w, hi_h = int(tw * 1.12), int(th * 1.12)
    src_hi = cover_resize(fallback_img, hi_w, hi_h) if fallback_img is not None else Image.new("RGB", (hi_w, hi_h), (20, 23, 30))

    for fi in range(n_frames):
        t_loc = fi / fps
        if windows:
            want_idx = len(windows) - 1
            for wi, (t0_w, t1_w, _) in enumerate(windows):
                if t_loc < t1_w - 1e-4:
                    want_idx = wi
                    break
            if want_idx != active_idx:
                if dec_proc is not None:
                    dec_proc.terminate()
                active_idx = want_idx
                cur_sub = windows[active_idx][2]
                dec_proc = _open_decoder(Path(cur_sub["mp4"]))
                active_badge = cur_sub.get("source_badge") or active_badge

        frame: Optional[Image.Image] = None
        if dec_proc is not None and dec_proc.stdout is not None:
            raw = dec_proc.stdout.read(frame_bytes)
            if len(raw) == frame_bytes:
                frame = Image.frombytes("RGB", (tw, th), raw)

        if frame is None:
            p = 0.5 - 0.5 * math.cos(math.pi * (fi / max(1, n_frames - 1)))
            cw = int(round(hi_w - (hi_w - tw) * p))
            ch = int(round(cw * th / tw))
            ox = (hi_w - cw) // 2
            oy = (hi_h - ch) // 2
            frame = src_hi.crop((ox, oy, ox + cw, oy + ch)).resize((tw, th), Image.Resampling.BILINEAR)

        draw = ImageDraw.Draw(frame, "RGBA")
        if not is_vertical:
            draw.rectangle([0, th - 210, tw, th], fill=(8, 10, 14, 92))
            hb = draw.textbbox((0, 0), hud_txt, font=f_hud)
            hx = tw - (hb[2] - hb[0]) - 84
            draw.rounded_rectangle([hx - 18, 52, tw - 64, 96], radius=22, fill=(12, 14, 18, 150), outline=(255, 255, 255, 50), width=1)
            draw.text((hx, 58), hud_txt, font=f_hud, fill=(238, 242, 248, 245))
            if show_news_card and t_loc >= 0.45:
                overlay_crawled_news_card(frame, beat)
            draw_source_badge(draw, active_badge, x=96, y=858, light_theme=False)
        else:
            draw.rectangle([0, th - 480, tw, th], fill=(8, 10, 14, 110))
            pill_txt = (beat.get("topic_pill") or "Hey Digital 選購結論")[:14]
            f_pill = get_cjk_font(26, bold=True)
            pb = draw.textbbox((0, 0), pill_txt, font=f_pill)
            draw.rounded_rectangle([62, 86, 62 + (pb[2] - pb[0]) + 64, 142], radius=28, fill=(14, 20, 16, 235))
            draw.ellipse([82, 106, 98, 122], fill=(46, 204, 113, 255))
            draw.text((110, 96), pill_txt, font=f_pill, fill=(255, 255, 255, 255))

        cap = caption_for_timestamp(beat, caption_timings, t_loc, real_dur)
        draw_adaptive_caption(frame, cap, is_light_bg=False, is_vertical=is_vertical)
        enc_proc.stdin.write(frame.tobytes())

    if dec_proc is not None:
        dec_proc.terminate()
    enc_proc.stdin.close()
    enc_proc.wait()


WEBGL_LONG_TYPES = {
    "webgl_memory_lanes",
    "webgl_stacked_tco",
    "webgl_ssd_particles",
    "webgl_chip_die",
    "webgl_dual_laptop",
    "ui_configurator_crawl",
    "studio_isometric_infographic",
}


def assemble_long_video(
    plan: dict,
    tts_dir: Path,
    img_dir: Path,
    out_mp4: Path,
    work_dir: Path,
    broll_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    fps: int = 30,
) -> dict:
    """Render V2 `1920x1080@30fps` Long-Form video with Three.js WebGL2, real B-roll/I2V, Rhodes BGM & Tactile SFX."""
    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = work_dir / "long_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    run_dir = run_dir or work_dir.parent

    tts_meta = json.loads((tts_dir / "tts_results.json").read_text(encoding="utf-8"))
    dur_map = {it["id"]: float(it["duration"]) for it in tts_meta["items"]}
    timings_map = {it["id"]: (it.get("caption_timings") or []) for it in tts_meta["items"]}
    silent_map = {it["id"]: bool(it.get("silent")) for it in tts_meta["items"]}

    harvested_mp4s = sorted(broll_dir.glob("BROLL_*.mp4")) if (broll_dir and broll_dir.exists()) else []
    broll_cursor = 0

    beats = plan.get("long_beats", [])
    segment_mp4s = []
    chapter_timestamps = []
    sfx_events: list[dict] = []
    seen_chapters = set()
    cur_time = 0.0

    with MotionStudioRenderer(repo_root=repo_root, run_dir=run_dir) as studio:
        for idx, beat in enumerate(beats):
            bid = beat["id"]
            wav_path = tts_dir / f"{bid}.wav"
            dur = dur_map.get(bid, float(beat.get("silent_duration") or 7.5))
            n_frames = max(int(fps * 1.2), int(round(dur * fps)))
            real_dur = n_frames / fps

            ch_num = str(beat.get("chapter_num") or "01").zfill(2)
            if ch_num not in seen_chapters:
                seen_chapters.add(ch_num)
                mm, ss = divmod(int(cur_time), 60)
                chapter_timestamps.append({
                    "time": f"{mm}:{ss:02d}",
                    "seconds": round(cur_time, 1),
                    "title": beat.get("chapter_title", ""),
                })

            bg_path = img_dir / f"{bid}.png"
            if not bg_path.exists():
                cands = sorted(img_dir.glob("L*.png"))
                bg_path = cands[idx % len(cands)] if cands else bg_path
            bg_img = Image.open(bg_path).convert("RGB") if bg_path.exists() else None

            vtype = beat.get("visual_type") or "cinema_broll_video"
            if vtype == "studio_isometric_infographic":
                vtype = "webgl_memory_lanes"
            seg_mp4 = clips_dir / f"{bid}.mp4"

            if vtype == "chapter_dark_card":
                sfx_events.append({"time": cur_time + 0.06, "kind": "whoosh", "pan": 0.0})
                ff_cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(fps), "-i", "-",
                    "-i", str(wav_path),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
                    "-t", f"{real_dur:.3f}",
                    str(seg_mp4),
                ]
                proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                assert proc.stdin is not None
                for fi in range(n_frames):
                    t_loc = fi / fps
                    prog = fi / max(1, n_frames - 1)
                    frame = render_chapter_dark_card_animated(beat, bg_img, prog)
                    if not silent_map.get(bid, True):
                        cap = caption_for_timestamp(beat, timings_map.get(bid, []), t_loc, real_dur)
                        draw_adaptive_caption(frame, cap, is_light_bg=False, is_vertical=False)
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
                proc.wait()

            elif vtype in WEBGL_LONG_TYPES:
                if os.environ.get("HEYDIGITAL_REUSE_WEBGL") == "1" and seg_mp4.exists() and seg_mp4.stat().st_size > 40_000:
                    sfx_events.append({"time": cur_time + 0.12, "kind": "pop", "pan": -0.15})
                    sfx_events.append({"time": cur_time + min(real_dur * 0.45, 2.2), "kind": "click", "pan": 0.15})
                else:
                    beat_enriched = dict(beat)
                    beat_enriched["visual_type"] = vtype
                    beat_enriched.setdefault("product_a", plan.get("product_a", "標準輕薄款"))
                    beat_enriched.setdefault("product_b", plan.get("product_b", "Pro 進階款"))
                    rel_sfx = studio.render_long_webgl_beat(
                        beat=beat_enriched,
                        wav_path=wav_path,
                        caption_timings=timings_map.get(bid, []),
                        real_dur=real_dur,
                        out_mp4=seg_mp4,
                        fps=fps,
                    )
                    for ev in rel_sfx:
                        sfx_events.append({"time": cur_time + ev["offset"], "kind": ev["kind"], "pan": ev.get("pan", 0.0)})

            else:
                # Cinema Motion B-Roll: prefer LTX-Video I2V clip -> harvested real HD B-roll clip -> high-res studio still
                i2v_cand = img_dir / f"{bid}_i2v.mp4"
                chosen_vid: Optional[Path] = None
                if i2v_cand.exists():
                    chosen_vid = i2v_cand
                elif harvested_mp4s:
                    chosen_vid = harvested_mp4s[broll_cursor % len(harvested_mp4s)]
                    broll_cursor += 1

                _render_broll_video_beat(
                    beat=beat,
                    wav_path=wav_path,
                    video_src=chosen_vid,
                    fallback_img=bg_img,
                    caption_timings=timings_map.get(bid, []),
                    real_dur=real_dur,
                    out_mp4=seg_mp4,
                    fps=fps,
                    is_vertical=False,
                    show_news_card=bool(beat.get("overlay_news_card")),
                )

            segment_mp4s.append(seg_mp4)
            cur_time += real_dur
            print(f"  [Compositor V2:Long] Rendered {bid} ({vtype}, {real_dur:.2f}s)")

    # Append 3.8s Public Crawled Sources Reference Slate
    sfx_events.append({"time": cur_time + 0.05, "kind": "whoosh", "pan": 0.0})
    slate_mp4 = clips_dir / "L99_sources.mp4"
    slate_img = render_sources_slate(plan.get("cited_sources", []))
    slate_frames = int(3.8 * fps)
    ff_slate = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(fps), "-i", "-",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-t", "3.8",
        str(slate_mp4),
    ]
    ps = subprocess.Popen(ff_slate, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert ps.stdin is not None
    raw_slate = slate_img.tobytes()
    for _ in range(slate_frames):
        ps.stdin.write(raw_slate)
    ps.stdin.close()
    ps.wait()
    segment_mp4s.append(slate_mp4)
    cur_time += 3.8

    concat_txt = work_dir / "long_concat.txt"
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in segment_mp4s) + "\n", encoding="utf-8")
    raw_joined = work_dir / "long_joined.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(raw_joined)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    bgm_wav = work_dir / "rhodes_chill_bgm.wav"
    sfx_wav = work_dir / "tactile_sfx.wav"
    generate_rhodes_chill_bgm(cur_time, bgm_wav, bpm=86.0)
    build_sfx_track(cur_time, sfx_events, sfx_wav)
    master_video_with_sidechain(raw_joined, bgm_wav, sfx_wav, out_mp4, target_lufs=-14.2)

    return {"duration": round(cur_time, 2), "chapters": chapter_timestamps, "path": str(out_mp4)}


def _render_cta_endcard_animated(
    beat: dict,
    wav_path: Path,
    thumb_img: Optional[Image.Image],
    caption_timings: list[dict],
    real_dur: float,
    out_mp4: Path,
    fps: int = 30,
):
    w, h = 1080, 1920
    n_frames = max(fps, int(round(real_dur * fps)))
    ff_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
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
        canvas = Image.new("RGB", (w, h), (12, 15, 20))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rounded_rectangle([58, 130, 1022, 1760], radius=46, fill=(20, 25, 34, 255), outline=(240, 202, 134, 150), width=3)
        draw.rounded_rectangle([106, 200, 476, 262], radius=30, fill=(22, 136, 76, 255))
        draw.text((136, 212), "● Hey Digital 完整長片已上線", font=get_cjk_font(24, True), fill=(255, 255, 255, 255))

        draw.text((106, 305), "想看完整實測曲線與", font=get_cjk_font(58, True), fill=(255, 255, 255, 255))
        draw.text((106, 388), "全部公開數據來源？", font=get_cjk_font(62, True), fill=(240, 202, 134, 255))

        if thumb_img is not None:
            zoom = 1.0 + 0.045 * min(1.0, t_loc / max(2.0, real_dur))
            tw_z, th_z = int(868 * zoom), int(488 * zoom)
            th_hi = cover_resize(thumb_img, tw_z, th_z)
            th_crop = th_hi.crop(((tw_z - 868) // 2, (th_z - 488) // 2, (tw_z - 868) // 2 + 868, (th_z - 488) // 2 + 488))
            mask = Image.new("L", (868, 488), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, 868, 488], radius=28, fill=255)
            canvas.paste(th_crop, (106, 515), mask)
            draw.rounded_rectangle([106, 515, 974, 1003], radius=28, outline=(255, 255, 255, 95), width=3)

        pulse = int(8 * math.sin(t_loc * 5.0))
        draw.rounded_rectangle([106 - pulse, 1085 - pulse // 2, 974 + pulse, 1215 + pulse // 2], radius=32, fill=(240, 202, 134, 255))
        draw.text((178, 1122), "▶ 點擊下方關聯影片看完整解析", font=get_cjk_font(44, True), fill=(15, 17, 22, 255))

        cap = caption_for_timestamp(beat, caption_timings, t_loc, real_dur)
        draw_adaptive_caption(canvas, cap, is_light_bg=False, is_vertical=True)
        proc.stdin.write(canvas.tobytes())

    proc.stdin.close()
    proc.wait()


def assemble_short_video(
    plan: dict,
    tts_dir: Path,
    img_dir: Path,
    thumb_path: Path,
    out_mp4: Path,
    work_dir: Path,
    broll_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    fps: int = 30,
) -> dict:
    """Render V2 `1080x1920@30fps` Interactive Traffic-Funnel Short with spring stickers, live iOS toggles & SFX."""
    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = work_dir / "short_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    run_dir = run_dir or work_dir.parent

    tts_meta = json.loads((tts_dir / "tts_results.json").read_text(encoding="utf-8"))
    dur_map = {it["id"]: float(it["duration"]) for it in tts_meta["items"]}
    timings_map = {it["id"]: (it.get("caption_timings") or []) for it in tts_meta["items"]}

    thumb_img = Image.open(thumb_path).convert("RGB") if thumb_path.exists() else None
    harvested_mp4s = sorted(broll_dir.glob("BROLL_*.mp4")) if (broll_dir and broll_dir.exists()) else []

    sbeats = plan.get("short_beats", [])
    step_beats = [b for b in sbeats if b.get("visual_type") in ("short_phone_interactive", "step_phone_mockup")]
    step_total = max(1, len(step_beats))

    # Prepare vertical 9:16 hook background clip from SigLIP-matched real B-roll (or I2V / harvested clip)
    hook_vert_rel = None
    s01_matched_mp4 = Path(sbeats[0]["matched_broll_mp4"]) if (sbeats and sbeats[0].get("matched_broll_mp4")) else None
    s01_i2v = img_dir / "S01_i2v.mp4"
    hook_src_mp4 = s01_matched_mp4 if (s01_matched_mp4 and s01_matched_mp4.exists()) else (harvested_mp4s[0] if harvested_mp4s else None)
    if hook_src_mp4 is not None:
        hv_out = work_dir / "hook_vert_bg.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(hook_src_mp4),
            "-an", "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,format=yuv420p",
            "-g", "15", "-c:v", "libx264", "-preset", "fast", "-crf", "20", str(hv_out),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if hv_out.exists():
            hook_vert_rel = str(hv_out.relative_to(run_dir))
    elif s01_i2v.exists():
        hook_vert_rel = str(s01_i2v.relative_to(run_dir))

    segment_mp4s = []
    sfx_events: list[dict] = []
    cur_step = 0
    total_dur = 0.0

    with MotionStudioRenderer(repo_root=repo_root, run_dir=run_dir) as studio:
        for idx, beat in enumerate(sbeats):
            bid = beat["id"]
            wav_path = tts_dir / f"{bid}.wav"
            dur = dur_map.get(bid, 7.0)
            n_frames = max(fps, int(round(dur * fps)))
            real_dur = n_frames / fps

            vtype = beat.get("visual_type") or "short_phone_interactive"
            if vtype == "hook_sticker_broll":
                vtype = "short_hook_dynamic"
            elif vtype == "step_phone_mockup":
                vtype = "short_phone_interactive"

            # Prefer SigLIP-matched real hardware keyframe inside Short phone steps!
            matched_jpg = Path(beat["matched_broll_jpg"]) if beat.get("matched_broll_jpg") else None
            bg_png = matched_jpg if (matched_jpg and matched_jpg.exists()) else (img_dir / f"{bid}.png")
            if not bg_png.exists():
                cands = sorted(list(img_dir.glob("*.png")))
                bg_png = cands[idx % len(cands)] if cands else bg_png
            bg_rel = str(bg_png.relative_to(run_dir)) if (bg_png.exists() and str(bg_png.resolve()).startswith(str(run_dir.resolve()))) else None
            bg_img = Image.open(bg_png).convert("RGB") if bg_png.exists() else None

            seg_mp4 = clips_dir / f"{bid}.mp4"

            if vtype in ("short_hook_dynamic", "short_phone_interactive"):
                if vtype == "short_phone_interactive":
                    cur_step += 1
                beat_copy = dict(beat)
                beat_copy["visual_type"] = vtype
                rel_sfx = studio.render_short_interactive_beat(
                    beat=beat_copy,
                    wav_path=wav_path,
                    caption_timings=timings_map.get(bid, []),
                    real_dur=real_dur,
                    out_mp4=seg_mp4,
                    step_idx=max(1, cur_step),
                    step_total=step_total,
                    bg_img_rel_run=bg_rel,
                    bg_vid_rel_run=hook_vert_rel if vtype == "short_hook_dynamic" else None,
                    fps=fps,
                )
                for ev in rel_sfx:
                    sfx_events.append({"time": total_dur + ev["offset"], "kind": ev["kind"], "pan": ev.get("pan", 0.0)})

            elif vtype == "short_broll_summary":
                matched_v = Path(beat["matched_broll_mp4"]) if beat.get("matched_broll_mp4") else None
                vid_cand = img_dir / f"{bid}_i2v.mp4"
                chosen_v = matched_v if (matched_v and matched_v.exists()) else (vid_cand if vid_cand.exists() else (harvested_mp4s[-1] if harvested_mp4s else None))
                _render_broll_video_beat(
                    beat=beat,
                    wav_path=wav_path,
                    video_src=chosen_v,
                    fallback_img=bg_img,
                    caption_timings=timings_map.get(bid, []),
                    real_dur=real_dur,
                    out_mp4=seg_mp4,
                    fps=fps,
                    is_vertical=True,
                )

            else:
                sfx_events.append({"time": total_dur + 0.06, "kind": "whoosh", "pan": 0.0})
                _render_cta_endcard_animated(
                    beat=beat,
                    wav_path=wav_path,
                    thumb_img=thumb_img,
                    caption_timings=timings_map.get(bid, []),
                    real_dur=real_dur,
                    out_mp4=seg_mp4,
                    fps=fps,
                )

            segment_mp4s.append(seg_mp4)
            total_dur += real_dur
            print(f"  [Compositor V2:Short] Rendered {bid} ({vtype}, {real_dur:.2f}s)")

    concat_txt = work_dir / "short_concat.txt"
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in segment_mp4s) + "\n", encoding="utf-8")
    raw_joined = work_dir / "short_joined.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(raw_joined)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    bgm_wav = work_dir / "short_rhodes_bgm.wav"
    sfx_wav = work_dir / "short_sfx.wav"
    generate_rhodes_chill_bgm(total_dur, bgm_wav, bpm=88.0)
    build_sfx_track(total_dur, sfx_events, sfx_wav)
    master_video_with_sidechain(raw_joined, bgm_wav, sfx_wav, out_mp4, target_lufs=-14.3, bgm_volume=0.14)

    return {"duration": round(total_dur, 2), "path": str(out_mp4)}
