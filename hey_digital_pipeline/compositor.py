#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon — 1:1 Studio Video, Traffic-Funnel Shorts & Thumbnail Compositor
=============================================================================================
Reconstructs the exact visual language of `@hey_digitall`:
- Long-Form (`1920x1080 @ 30fps`):
  1. `chapter_dark_card`: Matte charcoal micro-grid + outlined giant chapter numeral + gold eyebrow
  2. `studio_isometric_infographic`: Warm `#ECEAE4` studio canvas + thin giant colored hero metric
     (`153 GB/s`, `13 tok/s`) + 3-face shaded isometric 3D comparison lanes/bars + crawled source badge
  3. `cinema_tech_broll`: Smooth Ken Burns motion on RTX Pro 6000 AI studio photography + optional
     floating frosted-glass live crawled news/benchmark card + top-right chapter HUD
  4. `adaptive_clean_caption`: Borderless crisp dark `#181818` text on cream scenes & `#FFFFFF`
     soft-shadow text on dark scenes, frame-synced to Alan's PyTorch cloned voiceover!
- Traffic-Funnel Short (`1080x1920 @ 30fps`):
  1. `hook_sticker_broll`: Vertical hero shot + tilted white rounded sticker banners + vector icon
  2. `step_phone_mockup`: `#DCEFD8 -> #F6FAF4` mint-cream gradient + `● Topic` black pill + `01 / 05`
     emerald step header + `① → ② → ③` breadcrumb + iPhone bezel mockup + yellow callout sticker + progress bar
  3. `cta_funnel_endcard`: Long-form thumbnail showcase + explicit funnel CTA to watch the full video
- Thumbnail (`1920x1080` PNG):
  Dark radial spotlight + 2-line ultra-bold white/champagne-gold `#F0CA86` headline + product showcase.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


CJK_FONT_CANDIDATES_BOLD = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
]

LATIN_THIN_CANDIDATES = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def get_cjk_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    for fp in CJK_FONT_CANDIDATES_BOLD:
        if os.path.exists(fp):
            try:
                # PingFang.ttc index 0=Regular, 2=Semibold/Medium on macOS
                idx = 2 if (bold and fp.endswith("PingFang.ttc")) else 0
                return ImageFont.truetype(fp, size=size, index=idx)
            except Exception:
                try:
                    return ImageFont.truetype(fp, size=size)
                except Exception:
                    continue
    return ImageFont.load_default()


def get_latin_thin_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in LATIN_THIN_CANDIDATES:
        if os.path.exists(fp):
            try:
                # HelveticaNeue.ttc index 7 or 3 is Light/Thin on macOS
                idx = 7 if fp.endswith("HelveticaNeue.ttc") else 0
                return ImageFont.truetype(fp, size=size, index=idx)
            except Exception:
                try:
                    return ImageFont.truetype(fp, size=size)
                except Exception:
                    continue
    return get_cjk_font(size, bold=False)


def hex_to_rgb(h: str, default: tuple[int, int, int] = (14, 110, 184)) -> tuple[int, int, int]:
    h = (h or "").strip().lstrip("#")
    if len(h) == 6:
        try:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        except ValueError:
            pass
    return default


def adjust_rgb(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)  # type: ignore


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
    if not text:
        return
    w, h = canvas.size
    font_size = 50 if is_vertical else 44
    font = get_cjk_font(font_size, bold=True)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Wrap if needed for vertical
    max_px = int(w * (0.86 if is_vertical else 0.84))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    if tw > max_px and len(text) > 10:
        mid = len(text) // 2
        lines = [text[:mid].strip(), text[mid:].strip()]
    else:
        lines = [text]

    base_y = int(h * (0.805 if is_vertical else 0.905)) - (len(lines) - 1) * (font_size + 12)
    for idx, ln in enumerate(lines):
        bb = draw.textbbox((0, 0), ln, font=font)
        lw = bb[2] - bb[0]
        x = (w - lw) // 2
        y = base_y + idx * (font_size + 12)
        if is_light_bg and not is_vertical:
            # Exact Hey Digital cream-infographic dark caption (#181818, subtle crispness, no box)
            draw.text((x, y + 1), ln, font=font, fill=(255, 255, 255, 130))
            draw.text((x, y), ln, font=font, fill=(22, 22, 24, 255))
        else:
            # Crisp white caption with clean dark stroke + soft drop shadow
            draw.text((x + 2, y + 3), ln, font=font, fill=(0, 0, 0, 175))
            draw.text(
                (x, y),
                ln,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=3 if is_vertical else 2,
                stroke_fill=(10, 12, 16, 235),
            )


def draw_source_badge(draw: ImageDraw.ImageDraw, badge_text: str, x: int, y: int, light_theme: bool):
    if not badge_text:
        return
    font = get_cjk_font(22, bold=False)
    bb = draw.textbbox((0, 0), badge_text, font=font)
    pw = (bb[2] - bb[0]) + 38
    ph = 40
    if light_theme:
        draw.rounded_rectangle([x, y, x + pw, y + ph], radius=20, fill=(255, 255, 255, 195), outline=(206, 202, 192, 255), width=1)
        draw.ellipse([x + 14, y + 15, x + 24, y + 25], fill=(14, 124, 71, 255))
        draw.text((x + 32, y + 7), badge_text, font=font, fill=(68, 67, 63, 255))
    else:
        draw.rounded_rectangle([x, y, x + pw, y + ph], radius=20, fill=(14, 16, 22, 185), outline=(255, 255, 255, 55), width=1)
        draw.ellipse([x + 14, y + 15, x + 24, y + 25], fill=(229, 153, 52, 255))
        draw.text((x + 32, y + 7), badge_text, font=font, fill=(228, 232, 238, 245))


def render_chapter_dark_card(beat: dict, bg_img: Optional[Image.Image] = None) -> Image.Image:
    w, h = 1920, 1080
    canvas = Image.new("RGB", (w, h), (11, 12, 15))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Subtle dot grid
    for gx in range(40, w, 36):
        for gy in range(40, h, 36):
            draw.point((gx, gy), fill=(36, 39, 46, 150))

    # Subtle bottom-right hardware render silhouette
    if bg_img is not None:
        sub = cover_resize(bg_img, 980, 620)
        sub = ImageEnhance.Brightness(sub).enhance(0.36)
        mask = Image.new("L", (980, 620), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle([40, 40, 960, 600], radius=36, fill=210)
        mask = mask.filter(ImageFilter.GaussianBlur(42))
        canvas.paste(sub, (920, 440), mask)

    draw = ImageDraw.Draw(canvas, "RGBA")
    ch_num = str(beat.get("chapter_num") or "01").zfill(2)
    ch_en = (beat.get("chapter_en") or "DEEP DIVE").upper()
    ch_title = beat.get("chapter_title") or "核心拆解"
    subtitle = (beat.get("narration") or "")[:22].strip("，。！？ ")

    # Giant outlined chapter numeral on left (exact match to D9IiQF_QIYg/t0360s.jpg!)
    f_giant = get_latin_thin_font(470)
    draw.text((48, 235), ch_num, font=f_giant, fill=(11, 12, 15, 0), stroke_width=2, stroke_fill=(130, 135, 145, 210))

    # Right column eyebrow + divider + title
    rx = 995
    f_en = get_cjk_font(23, bold=True)
    spaced_en = " ".join(list(f"CHAPTER {ch_num} / {ch_en}"))
    draw.text((rx, 345), spaced_en[:56], font=f_en, fill=(229, 153, 52, 255))
    draw.line([(rx, 396), (1770, 396)], fill=(66, 70, 78, 255), width=2)

    f_title = get_cjk_font(68, bold=True)
    draw.text((rx, 430), ch_title, font=f_title, fill=(255, 255, 255, 255))

    f_sub = get_cjk_font(30, bold=False)
    draw.text((rx, 530), subtitle, font=f_sub, fill=(140, 145, 154, 255))
    return canvas


def _draw_iso_bar_lane(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    total_blocks: int,
    active_blocks: int,
    accent_rgb: tuple[int, int, int],
    label_left: str,
    label_right: str,
):
    """Draw Hey Digital's signature 3D isometric chip/memory block lane on warm studio backdrop."""
    # Rail base
    rail_len = total_blocks * 48 + 36
    dx_step = 46
    dy_step = 5
    # Back rail shadow & bar
    pts_rail = [
        (x0 - 16, y0 + 36),
        (x0 + rail_len, y0 + 36 + int(rail_len * 0.11)),
        (x0 + rail_len - 10, y0 + 50 + int(rail_len * 0.11)),
        (x0 - 26, y0 + 50),
    ]
    draw.polygon(pts_rail, fill=(198, 194, 184, 255))

    top_active = adjust_rgb(accent_rgb, 1.18)
    front_active = accent_rgb
    side_active = adjust_rgb(accent_rgb, 0.76)

    top_idle = (214, 211, 204)
    front_idle = (192, 189, 181)
    side_idle = (166, 163, 155)

    bw, bh, bd = 30, 34, 14
    for i in range(total_blocks):
        bx = x0 + i * dx_step
        by = y0 + i * dy_step
        is_on = i < active_blocks
        c_top = top_active if is_on else top_idle
        c_front = front_active if is_on else front_idle
        c_side = side_active if is_on else side_idle

        # Soft shadow
        draw.polygon(
            [(bx - 4, by + bh + 4), (bx + bw + 8, by + bh + 6), (bx + bw + 2, by + bh + 12), (bx - 8, by + bh + 10)],
            fill=(170, 166, 156, 120),
        )
        # Front face
        draw.polygon([(bx, by), (bx + bw, by + 3), (bx + bw, by + bh + 3), (bx, by + bh)], fill=c_front)
        # Top face
        draw.polygon([(bx, by), (bx + bd, by - 10), (bx + bw + bd, by - 7), (bx + bw, by + 3)], fill=c_top)
        # Right side face
        draw.polygon([(bx + bw, by + 3), (bx + bw + bd, by - 7), (bx + bw + bd, by + bh - 7), (bx + bw, by + bh + 3)], fill=c_side)

    f_lbl = get_cjk_font(25, bold=True)
    f_val = get_cjk_font(26, bold=True)
    draw.text((x0 - 240, y0 + 2), label_left[:14], font=f_lbl, fill=(58, 57, 53, 255))
    end_x = x0 + rail_len + 24
    end_y = y0 + int(rail_len * 0.11) + 4
    draw.text((end_x, end_y), label_right[:18], font=f_val, fill=accent_rgb if active_blocks > 0 else (90, 89, 85))


def render_studio_infographic_base(beat: dict, progress: float = 1.0) -> Image.Image:
    """Render Family B: Warm #ECEAE4 studio 3D isometric infographic with hero metric & source citation."""
    w, h = 1920, 1080
    # Radial warm cream gradient (#F5F3EE center -> #E7E4DC edges)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - w * 0.52) / (w * 0.7)) ** 2 + ((yy - h * 0.45) / (h * 0.7)) ** 2)
    dist = np.clip(dist, 0.0, 1.0)
    r = (245 - dist * 15).astype(np.uint8)
    g = (243 - dist * 16).astype(np.uint8)
    b = (238 - dist * 18).astype(np.uint8)
    canvas = Image.fromarray(np.dstack([r, g, b]), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")

    accent = hex_to_rgb(beat.get("accent_hex") or "#0E6EB8")

    # Top-right section HUD (`02 · 記憶體刀法`)
    ch_num = str(beat.get("chapter_num") or "01").zfill(2)
    ch_title = beat.get("chapter_title") or "數據實測"
    hud_txt = f"{ch_num}  ·  {ch_title}"
    f_hud = get_cjk_font(24, bold=False)
    hb = draw.textbbox((0, 0), hud_txt, font=f_hud)
    draw.text((w - (hb[2] - hb[0]) - 86, 64), hud_txt, font=f_hud, fill=(92, 90, 84, 255))

    # Top-left Giant Thin Hero Metric (`153 GB/s`)
    metric_val = str(beat.get("hero_metric") or "4.8x")
    metric_unit = str(beat.get("hero_unit") or "")
    f_hero = get_latin_thin_font(196)
    f_unit = get_latin_thin_font(60)
    draw.text((112, 82), metric_val, font=f_hero, fill=accent)
    vb = draw.textbbox((112, 82), metric_val, font=f_hero)
    if metric_unit:
        draw.text((vb[2] + 18, vb[3] - 68), metric_unit, font=f_unit, fill=(106, 104, 98, 255))

    # Subtitle context under hero metric
    src_detail = beat.get("source_detail") or {}
    sub_note = src_detail.get("title", "")[:34] if src_detail else ch_title
    if sub_note:
        f_sub = get_cjk_font(24, bold=False)
        draw.text((118, 305), sub_note, font=f_sub, fill=(116, 114, 108, 235))

    # Isometric 3D comparison block lanes
    bars = beat.get("comparison_bars") or []
    if not bars:
        bars = [
            {"label": "旗艦高配規格", "value": 92, "display": f"{metric_val} {metric_unit}".strip()},
            {"label": "主流標準規格", "value": 64, "display": "標準基準"},
            {"label": "入門基礎版本", "value": 38, "display": "頻寬受限"},
        ]
    max_val = max(float(x.get("value") or 1) for x in bars) or 100.0
    palette = [accent, (14, 110, 184), (184, 74, 20), (14, 124, 71)]

    start_y = 405 if len(bars) <= 3 else 365
    gap_y = 138 if len(bars) <= 3 else 112
    total_blocks = 16
    for idx, bar in enumerate(bars[:4]):
        ratio = max(0.08, min(1.0, (float(bar.get("value") or 50) / max_val) * progress))
        act = max(1, int(round(total_blocks * ratio)))
        col = palette[idx % len(palette)] if idx == 0 else adjust_rgb(palette[idx % len(palette)], 0.92)
        _draw_iso_bar_lane(
            draw,
            x0=610,
            y0=start_y + idx * gap_y,
            total_blocks=total_blocks,
            active_blocks=act,
            accent_rgb=col,
            label_left=str(bar.get("label") or f"組別 {idx+1}"),
            label_right=str(bar.get("display") or f"{bar.get('value')}"),
        )

    # Crawled Source Citation Badge (Bottom-Left)
    badge = beat.get("source_badge") or ""
    sid = beat.get("source_id") or ""
    if badge and sid and sid not in badge:
        badge = f"{badge} · {sid}"
    draw_source_badge(draw, badge, x=112, y=855, light_theme=True)
    return canvas


def overlay_crawled_news_card(canvas: Image.Image, beat: dict):
    """Overlay a sleek floating frosted-glass webpage/spec card (replicating l0abQQ4U0YA's 9to5mac card!)."""
    src = beat.get("source_detail")
    if not src or not beat.get("overlay_news_card"):
        return
    draw = ImageDraw.Draw(canvas, "RGBA")
    cx, cy, cw, ch = 118, 565, 650, 255
    # Frosted dark-glass card
    region = canvas.crop((cx, cy, cx + cw, cy + ch)).filter(ImageFilter.GaussianBlur(14))
    region = ImageEnhance.Brightness(region).enhance(0.45)
    mask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, cw, ch], radius=24, fill=235)
    canvas.paste(region, (cx, cy), mask)

    draw.rounded_rectangle([cx, cy, cx + cw, cy + ch], radius=24, outline=(255, 255, 255, 75), width=2)
    # Browser chrome dots + domain bar
    draw.ellipse([cx + 22, cy + 20, cx + 34, cy + 32], fill=(255, 95, 86, 230))
    draw.ellipse([cx + 42, cy + 20, cx + 54, cy + 32], fill=(255, 189, 46, 230))
    draw.ellipse([cx + 62, cy + 20, cx + 74, cy + 32], fill=(39, 201, 63, 230))

    f_dom = get_cjk_font(20, bold=True)
    domain_str = f"{src.get('publisher', 'Public Web')}  ·  {src.get('domain', '')}  ({src.get('published', '')[:10]})"
    draw.text((cx + 92, cy + 14), domain_str[:46], font=f_dom, fill=(185, 195, 210, 245))
    draw.line([(cx + 20, cy + 48), (cx + cw - 20, cy + 48)], fill=(255, 255, 255, 40), width=1)

    f_t = get_cjk_font(26, bold=True)
    title = src.get("title", "")
    line1 = title[:24]
    line2 = title[24:48] + ("…" if len(title) > 48 else "")
    draw.text((cx + 24, cy + 64), line1, font=f_t, fill=(255, 255, 255, 255))
    if line2:
        draw.text((cx + 24, cy + 100), line2, font=f_t, fill=(255, 255, 255, 255))

    metrics = src.get("extracted_metrics") or []
    f_m = get_cjk_font(21, bold=False)
    snip = ("關鍵數據：" + " / ".join(metrics[:4])) if metrics else (src.get("summary", "")[:52] + "…")
    draw.text((cx + 24, cy + 152), snip[:36], font=f_m, fill=(240, 202, 134, 255))
    draw.text((cx + 24, cy + 198), f"公開來源編號 {src.get('source_id', '')} · 已驗證收錄於資訊欄", font=get_cjk_font(18, False), fill=(155, 165, 180, 230))


def render_sources_slate(cited_sources: list[dict]) -> Image.Image:
    """Render clean end-of-video journalistic public sources summary slate."""
    w, h = 1920, 1080
    canvas = Image.new("RGB", (w, h), (13, 15, 19))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.text((110, 82), "PUBLIC DATA & CRAWLED SOURCES", font=get_cjk_font(22, True), fill=(229, 153, 52, 255))
    draw.text((110, 118), "本期影片公開爬蟲資訊與參考來源", font=get_cjk_font(48, True), fill=(255, 255, 255, 255))
    draw.line([(110, 192), (1810, 192)], fill=(55, 60, 70, 255), width=2)

    f_id = get_cjk_font(22, True)
    f_tt = get_cjk_font(25, True)
    f_url = get_cjk_font(19, False)
    for idx, s in enumerate(cited_sources[:6]):
        y = 225 + idx * 122
        draw.rounded_rectangle([110, y, 1810, y + 104], radius=16, fill=(22, 25, 32, 255), outline=(45, 50, 62, 255), width=1)
        draw.rounded_rectangle([132, y + 20, 238, y + 56], radius=10, fill=(229, 153, 52, 40), outline=(229, 153, 52, 190), width=1)
        draw.text((148, y + 24), s.get("source_id", f"SRC-{idx+1:02d}"), font=f_id, fill=(240, 202, 134, 255))
        draw.text((260, y + 18), f"{s.get('publisher', '')} — {s.get('title', '')[:54]}", font=f_tt, fill=(242, 245, 250, 255))
        draw.text((260, y + 60), s.get("url", "")[:95], font=f_url, fill=(138, 148, 165, 255))
    return canvas


def render_short_frame(
    beat: dict,
    bg_img: Optional[Image.Image],
    thumb_img: Optional[Image.Image],
    caption_text: str,
    step_idx: int,
    step_total: int,
    beat_progress: float,
) -> Image.Image:
    """Render 1080x1920 @ 30fps Short frame matching @hey_digitall's exact mint-step & sticker funnel style."""
    w, h = 1080, 1920
    vtype = beat.get("visual_type") or "step_phone_mockup"

    if vtype == "hook_sticker_broll":
        if bg_img is not None:
            # Subtle zoom
            z = 1.0 + 0.06 * beat_progress
            zw, zh = int(w * z), int(h * z)
            big = cover_resize(bg_img, zw, zh)
            canvas = big.crop(((zw - w) // 2, (zh - h) // 2, (zw - w) // 2 + w, (zh - h) // 2 + h))
        else:
            canvas = Image.new("RGB", (w, h), (20, 24, 30))
        draw = ImageDraw.Draw(canvas, "RGBA")

        # Top & secondary white rounded sticker cards (1:1 match to shorts_QrBoNy0wse8/t0003s.jpg!)
        st1 = beat.get("sticker_top") or "參數拉滿卻踩雷？"
        st2 = beat.get("sticker_bottom") or "真相差很大！"
        for text_s, (sx, sy, rot) in [(st1, (150, 215, 2.5)), (st2, (485, 430, -3.0))]:
            f_st = get_cjk_font(62, bold=True)
            tb = draw.textbbox((0, 0), text_s, font=f_st)
            sw, sh = (tb[2] - tb[0]) + 76, (tb[3] - tb[1]) + 54
            sticker = Image.new("RGBA", (sw + 24, sh + 24), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sticker, "RGBA")
            sd.rounded_rectangle([12, 14, sw + 12, sh + 14], radius=30, fill=(0, 0, 0, 70))
            sd.rounded_rectangle([8, 8, sw + 8, sh + 8], radius=30, fill=(255, 255, 255, 250))
            sd.text((46, 24), text_s, font=f_st, fill=(16, 18, 22, 255))
            rot_st = sticker.rotate(rot, resample=Image.Resampling.BICUBIC, expand=True)
            canvas.paste(rot_st, (sx, sy), rot_st)

        # Magnifier vector die-cut sticker on left
        icon = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
        idraw = ImageDraw.Draw(icon, "RGBA")
        idraw.ellipse([24, 24, 168, 168], fill=(110, 205, 248, 255), outline=(255, 255, 255, 255), width=14)
        idraw.ellipse([32, 32, 160, 160], outline=(22, 24, 28, 255), width=7)
        idraw.line([(152, 152), (212, 206)], fill=(255, 255, 255, 255), width=32)
        idraw.line([(152, 152), (212, 206)], fill=(238, 102, 45, 255), width=20)
        canvas.paste(icon, (125, 420), icon)

        draw_adaptive_caption(canvas, caption_text, is_light_bg=False, is_vertical=True)
        return canvas

    if vtype == "cta_funnel_endcard":
        canvas = Image.new("RGB", (w, h), (12, 15, 20))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rounded_rectangle([64, 150, 1016, 1740], radius=44, fill=(20, 25, 34, 255), outline=(240, 202, 134, 140), width=3)
        draw.rounded_rectangle([110, 220, 460, 280], radius=30, fill=(22, 136, 76, 255))
        draw.text((140, 232), "● Hey Digital 完整長片已上線", font=get_cjk_font(24, True), fill=(255, 255, 255, 255))

        draw.text((110, 325), "想看完整實測與", font=get_cjk_font(62, True), fill=(255, 255, 255, 255))
        draw.text((110, 410), "公開數據來源？", font=get_cjk_font(66, True), fill=(240, 202, 134, 255))

        if thumb_img is not None:
            th = cover_resize(thumb_img, 860, 484)
            mask = Image.new("L", (860, 484), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, 860, 484], radius=28, fill=255)
            canvas.paste(th, (110, 540), mask)
            draw.rounded_rectangle([110, 540, 970, 1024], radius=28, outline=(255, 255, 255, 90), width=3)

        draw.rounded_rectangle([110, 1100, 970, 1230], radius=32, fill=(240, 202, 134, 255))
        draw.text((168, 1136), "👇 點擊下方關聯影片看完整解析", font=get_cjk_font(44, True), fill=(15, 17, 22, 255))
        draw_adaptive_caption(canvas, caption_text, is_light_bg=False, is_vertical=True)
        return canvas

    # Default: Signature Mint-Cream Step Phone Mockup (`QrBoNy0wse8` 1:1 Replica!)
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    r = np.clip(222 + 24 * np.sin(yy * math.pi) - 8 * xx, 205, 250).astype(np.uint8)
    g = np.clip(242 + 10 * np.sin(yy * math.pi), 232, 253).astype(np.uint8)
    b = np.clip(218 + 26 * xx + 12 * np.sin(yy * math.pi), 210, 248).astype(np.uint8)
    canvas = Image.fromarray(np.dstack([r, g, b]), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Top-left black pill + green dot
    pill_txt = (beat.get("topic_pill") or "Hey Digital 實測避坑")[:14]
    f_pill = get_cjk_font(26, bold=True)
    pb = draw.textbbox((0, 0), pill_txt, font=f_pill)
    pw = (pb[2] - pb[0]) + 64
    draw.rounded_rectangle([64, 96, 64 + pw, 152], radius=28, fill=(15, 22, 18, 255))
    draw.ellipse([84, 116, 100, 132], fill=(46, 204, 113, 255))
    draw.text((112, 106), pill_txt, font=f_pill, fill=(255, 255, 255, 255))

    # Top-right counter `01 / 05`
    s_idx = max(1, step_idx)
    s_tot = max(1, step_total)
    draw.text((885, 104), f"{s_idx:02d}", font=get_cjk_font(30, True), fill=(18, 22, 18, 255))
    draw.text((932, 104), f"/ {s_tot:02d}", font=get_cjk_font(30, True), fill=(142, 152, 142, 255))

    # Giant Emerald Step Number + Bold Title
    draw.text((64, 172), f"{s_idx:02d}", font=get_cjk_font(126, True), fill=(24, 138, 76, 255))
    step_title = (beat.get("step_title") or "關鍵硬體差異")[:9]
    draw.text((262, 198), step_title, font=get_cjk_font(82, True), fill=(16, 18, 16, 255))

    # Breadcrumb flow `① ... → ② ... → ③ ...`
    flows = beat.get("flow_steps") or ["公開規格", "實測對比", "選購結論"]
    fx = 68
    f_fl = get_cjk_font(26, bold=True)
    circles = ["①", "②", "③", "④"]
    for i, ft in enumerate(flows[:3]):
        col = (24, 138, 76, 255) if i == min(2, int(beat_progress * 3)) else (32, 40, 34, 235)
        txt = f"{circles[i]} {ft[:8]}"
        draw.text((fx, 336), txt, font=f_fl, fill=col)
        tb = draw.textbbox((fx, 336), txt, font=f_fl)
        fx = tb[2] + 18
        if i < len(flows[:3]) - 1:
            draw.text((fx, 336), "→", font=f_fl, fill=(150, 168, 152, 255))
            fx += 38

    # Central Smartphone Bezel Mockup (x=220..860, y=420..1680)
    px0, py0, px1, py1 = 215, 420, 865, 1685
    draw.rounded_rectangle([px0 - 8, py0 + 12, px1 + 8, py1 + 18], radius=78, fill=(0, 0, 0, 45))
    draw.rounded_rectangle([px0, py0, px1, py1], radius=74, fill=(14, 18, 16, 255), outline=(35, 115, 72, 255), width=5)

    # Inner screen (x=242..838, y=448..1656)
    sx0, sy0, sx1, sy1 = px0 + 26, py0 + 26, px1 - 26, py1 - 26
    draw.rounded_rectangle([sx0, sy0, sx1, sy1], radius=54, fill=(246, 248, 250, 255))

    # Dynamic Island + Status bar
    draw.rounded_rectangle([462, sy0 + 18, 618, sy0 + 56], radius=19, fill=(10, 12, 14, 255))
    draw.text((sx0 + 34, sy0 + 22), "9:41", font=get_cjk_font(22, True), fill=(20, 22, 26, 255))
    draw.text((sx1 - 95, sy0 + 22), "5G 100%", font=get_cjk_font(19, True), fill=(20, 22, 26, 255))

    # Top half inside phone: AI hardware B-roll image
    if bg_img is not None:
        p_img = cover_resize(bg_img, sx1 - sx0 - 36, 430)
        pmask = Image.new("L", (sx1 - sx0 - 36, 430), 0)
        ImageDraw.Draw(pmask).rounded_rectangle([0, 0, sx1 - sx0 - 36, 430], radius=28, fill=255)
        canvas.paste(p_img, (sx0 + 18, sy0 + 82), pmask)

    # Bottom half inside phone: Crawled Metric & Key Points iOS-style grouped cards
    metric_str = str(beat.get("card_metric") or step_title)
    draw.rounded_rectangle([sx0 + 22, sy0 + 532, sx1 - 22, sy0 + 648], radius=24, fill=(255, 255, 255, 255), outline=(226, 232, 238, 255), width=2)
    draw.text((sx0 + 44, sy0 + 548), "核心實測指標", font=get_cjk_font(20, False), fill=(115, 122, 130, 255))
    draw.text((sx0 + 44, sy0 + 582), metric_str[:18], font=get_cjk_font(36, True), fill=(24, 138, 76, 255))

    pts = beat.get("card_points") or [beat.get("narration", "")[:20]]
    for pi, pt in enumerate(pts[:3]):
        cy = sy0 + 668 + pi * 118
        draw.rounded_rectangle([sx0 + 22, cy, sx1 - 22, cy + 102], radius=22, fill=(255, 255, 255, 255), outline=(230, 234, 240, 255), width=2)
        draw.ellipse([sx0 + 44, cy + 38, sx0 + 66, cy + 60], fill=(24, 138, 76, 255))
        draw.text((sx0 + 82, cy + 30), str(pt)[:18], font=get_cjk_font(26, True), fill=(28, 32, 36, 255))

    # Source citation footer inside phone screen
    src = beat.get("source_detail")
    if src:
        draw.text((sx0 + 36, sy1 - 54), f"來源：{src.get('publisher', '')[:18]} ({src.get('source_id', '')})", font=get_cjk_font(19, False), fill=(110, 118, 126, 255))

    # Side Yellow Callout Sticker (`快人一步` style from QrBoNy0wse8/t0145s.jpg!)
    tag = (beat.get("callout_tag") or "關鍵差異")[:6]
    st_im = Image.new("RGBA", (220, 96), (0, 0, 0, 0))
    sd = ImageDraw.Draw(st_im, "RGBA")
    sd.rounded_rectangle([6, 8, 212, 86], radius=20, fill=(250, 204, 42, 255), outline=(255, 255, 255, 255), width=5)
    sd.text((28, 22), tag, font=get_cjk_font(34, True), fill=(20, 20, 18, 255))
    st_rot = st_im.rotate(5.0, resample=Image.Resampling.BICUBIC, expand=True)
    canvas.paste(st_rot, (36, 720), st_rot)

    # Segmented Bottom Progress Bar (`y=1848`)
    seg_gap = 12
    seg_w = (w - 128 - seg_gap * (s_tot - 1)) // s_tot
    for si in range(s_tot):
        bx = 64 + si * (seg_w + seg_gap)
        draw.rounded_rectangle([bx, 1848, bx + seg_w, 1860], radius=6, fill=(192, 224, 200, 255))
        if si < s_idx - 1:
            draw.rounded_rectangle([bx, 1848, bx + seg_w, 1860], radius=6, fill=(24, 138, 76, 255))
        elif si == s_idx - 1:
            pw_cur = max(12, int(seg_w * beat_progress))
            draw.rounded_rectangle([bx, 1848, bx + pw_cur, 1860], radius=6, fill=(24, 138, 76, 255))

    draw_adaptive_caption(canvas, caption_text, is_light_bg=False, is_vertical=True)
    return canvas


def render_thumbnail(plan: dict, hero_img: Optional[Image.Image], out_path: Path) -> Path:
    """Render 1920x1080 studio spotlight thumbnail with white + champagne-gold (#F0CA86) bold typography."""
    w, h = 1920, 1080
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - w * 0.5) / (w * 0.62)) ** 2 + ((yy - h * 0.68) / (h * 0.58)) ** 2)
    dist = np.clip(dist, 0.0, 1.0)
    c_r = (34 * (1 - dist) + 8 * dist).astype(np.uint8)
    c_g = (38 * (1 - dist) + 9 * dist).astype(np.uint8)
    c_b = (46 * (1 - dist) + 12 * dist).astype(np.uint8)
    canvas = Image.fromarray(np.dstack([c_r, c_g, c_b]), mode="RGB")

    if hero_img is not None:
        prod = cover_resize(hero_img, 1320, 660)
        mask = Image.new("L", (1320, 660), 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle([30, 20, 1290, 650], radius=48, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(24))
        canvas.paste(prod, ((w - 1320) // 2, 415), mask)

    # Top vignette for crisp headline contrast
    grad = Image.new("RGBA", (w, 470), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad, "RGBA")
    for gy in range(470):
        alpha = int(215 * max(0.0, 1.0 - (gy / 460.0) ** 1.4))
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
        f_big = get_cjk_font(128 if len(ln) <= 11 else 108, bold=True)
        bb = draw.textbbox((0, 0), ln, font=f_big)
        tw = bb[2] - bb[0]
        x = (w - tw) // 2
        y = 64 + li * 158
        # Draw shadow
        draw.text((x + 4, y + 6), ln, font=f_big, fill=(0, 0, 0, 210))
        if gold and gold in ln:
            pre, post = ln.split(gold, 1)
            cx = x
            if pre:
                draw.text((cx, y), pre, font=f_big, fill=(255, 255, 255, 255))
                cx = draw.textbbox((cx, y), pre, font=f_big)[2]
            draw.text((cx, y), gold, font=f_big, fill=(240, 202, 134, 255))
            cx = draw.textbbox((cx, y), gold, font=f_big)[2]
            if post:
                draw.text((cx, y), post, font=f_big, fill=(255, 255, 255, 255))
        else:
            draw.text((x, y), ln, font=f_big, fill=(255, 255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path), quality=95)
    return out_path


def generate_ambient_bgm(duration_sec: float, out_wav: Path, sample_rate: int = 24000):
    """Synthesize warm, copyright-free minimalist studio ambient pad (-30dB bed like @hey_digitall)."""
    n = int((duration_sec + 1.5) * sample_rate)
    t = np.linspace(0, duration_sec + 1.5, n, endpoint=False, dtype=np.float32)
    # Warm Cmaj9 / Fmaj7 soft sine pad with slow LFO movement
    freqs = [130.81, 196.00, 246.94, 293.66, 392.00]
    sig_l = np.zeros_like(t)
    sig_r = np.zeros_like(t)
    for i, f in enumerate(freqs):
        lfo = 0.65 + 0.35 * np.sin(2 * math.pi * (0.07 + i * 0.019) * t + i)
        sig_l += (0.20 / (i + 1)) * np.sin(2 * math.pi * f * t) * lfo
        sig_r += (0.20 / (i + 1)) * np.sin(2 * math.pi * (f * 1.0015) * t + 0.6) * lfo

    # Gentle fade-in and fade-out
    fade_in = np.minimum(1.0, t / 2.0)
    fade_out = np.minimum(1.0, np.maximum(0.0, (duration_sec + 1.2 - t) / 2.5))
    env = fade_in * fade_out * 0.042  # subtle -28dB ambient studio bed
    stereo = np.stack([sig_l * env, sig_r * env], axis=-1)
    pcm = np.clip(stereo * 32767.0, -32767, 32767).astype(np.int16)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _caption_at_time(captions: list[str], t_local: float, dur: float) -> str:
    if not captions:
        return ""
    total_chars = sum(max(1, len(c)) for c in captions)
    acc = 0.0
    for c in captions:
        span = dur * (max(1, len(c)) / total_chars)
        if t_local <= acc + span + 1e-4:
            return c
        acc += span
    return captions[-1]


def assemble_long_video(
    plan: dict,
    tts_dir: Path,
    img_dir: Path,
    out_mp4: Path,
    work_dir: Path,
    fps: int = 30,
) -> dict:
    """Render frame-accurate 1920x1080@30fps Long-form video + return exact chapter timestamps."""
    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = work_dir / "long_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    tts_meta = json.loads((tts_dir / "tts_results.json").read_text(encoding="utf-8"))
    dur_map = {it["id"]: float(it["duration"]) for it in tts_meta["items"]}

    beats = plan.get("long_beats", [])
    segment_mp4s = []
    chapter_timestamps = []
    seen_chapters = set()
    cur_time = 0.0

    for idx, beat in enumerate(beats):
        bid = beat["id"]
        wav_path = tts_dir / f"{bid}.wav"
        dur = dur_map.get(bid, 8.0)
        n_frames = max(fps, int(round(dur * fps)))
        real_dur = n_frames / fps

        ch_num = str(beat.get("chapter_num") or "01").zfill(2)
        if ch_num not in seen_chapters:
            seen_chapters.add(ch_num)
            mm, ss = divmod(int(cur_time), 60)
            chapter_timestamps.append({"time": f"{mm}:{ss:02d}", "seconds": round(cur_time, 1), "title": beat.get("chapter_title", "")})

        bg_path = img_dir / f"{bid}.png"
        if not bg_path.exists():
            # Fallback to any available generated B-roll image
            cands = sorted(img_dir.glob("L*.png"))
            bg_path = cands[idx % len(cands)] if cands else bg_path
        bg_img = Image.open(bg_path).convert("RGB") if bg_path.exists() else None

        vtype = beat.get("visual_type") or "cinema_tech_broll"
        seg_mp4 = clips_dir / f"{bid}.mp4"

        # Stream raw RGB24 frames directly to FFmpeg H.264 encoder (fast, zero temp-PNG disk bloat!)
        ff_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(fps), "-i", "-",
            "-i", str(wav_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-t", f"{real_dur:.3f}",
            str(seg_mp4),
        ]
        proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None

        # Pre-cache static layers where possible for ultra-fast rendering
        if vtype == "chapter_dark_card":
            static_base = render_chapter_dark_card(beat, bg_img)
            for fi in range(n_frames):
                t_loc = fi / fps
                frame = static_base.copy()
                cap = _caption_at_time(beat.get("captions", []), t_loc, real_dur)
                draw_adaptive_caption(frame, cap, is_light_bg=False, is_vertical=False)
                proc.stdin.write(frame.tobytes())
        elif vtype == "studio_isometric_infographic":
            # Animate bar growth over first 1.1s, then hold steady
            cached_full = None
            for fi in range(n_frames):
                t_loc = fi / fps
                prog = min(1.0, 0.25 + 0.75 * (t_loc / 1.1))
                if prog >= 0.999:
                    if cached_full is None:
                        cached_full = render_studio_infographic_base(beat, progress=1.0)
                    frame = cached_full.copy()
                else:
                    frame = render_studio_infographic_base(beat, progress=prog)
                cap = _caption_at_time(beat.get("captions", []), t_loc, real_dur)
                draw_adaptive_caption(frame, cap, is_light_bg=True, is_vertical=False)
                proc.stdin.write(frame.tobytes())
        else:
            # Cinema Tech B-Roll with smooth Ken Burns zoom/pan + HUD + crawled card
            src_hi = cover_resize(bg_img, 2160, 1215) if bg_img is not None else Image.new("RGB", (2160, 1215), (18, 20, 26))
            zoom_in = (idx % 2 == 0)
            hud_txt = f"{ch_num}  ·  {beat.get('chapter_title', '')}"
            f_hud = get_cjk_font(24, bold=False)
            badge_txt = beat.get("source_badge") or ""
            sid = beat.get("source_id") or ""
            if badge_txt and sid and sid not in badge_txt:
                badge_txt = f"{badge_txt} · {sid}"

            for fi in range(n_frames):
                t_loc = fi / fps
                p = fi / max(1, n_frames - 1)
                ratio = p if zoom_in else (1.0 - p)
                cw = int(round(2160 - 200 * ratio))
                ch = int(round(cw * 9 / 16))
                ox = (2160 - cw) // 2
                oy = (1215 - ch) // 2
                frame = src_hi.crop((ox, oy, ox + cw, oy + ch)).resize((1920, 1080), Image.Resampling.BILINEAR)
                draw = ImageDraw.Draw(frame, "RGBA")
                hb = draw.textbbox((0, 0), hud_txt, font=f_hud)
                draw.text((1920 - (hb[2] - hb[0]) - 84, 62), hud_txt, font=f_hud, fill=(230, 234, 240, 235), stroke_width=2, stroke_fill=(0, 0, 0, 140))
                if beat.get("overlay_news_card") and t_loc >= 0.4:
                    overlay_crawled_news_card(frame, beat)
                draw_source_badge(ImageDraw.Draw(frame, "RGBA"), badge_txt, x=96, y=858, light_theme=False)
                cap = _caption_at_time(beat.get("captions", []), t_loc, real_dur)
                draw_adaptive_caption(frame, cap, is_light_bg=False, is_vertical=False)
                proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.wait()
        segment_mp4s.append(seg_mp4)
        cur_time += real_dur
        print(f"  [Compositor:Long] Rendered {bid} ({vtype}, {real_dur:.2f}s)")

    # Append 3.8s Public Crawled Sources Reference Slate at the end
    slate_mp4 = clips_dir / "L99_sources.mp4"
    slate_img = render_sources_slate(plan.get("cited_sources", []))
    slate_frames = int(3.8 * fps)
    ff_slate = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(fps), "-i", "-",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
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

    # Concatenate all segments + mix subtle studio ambient pad + master to -14.2 LUFS
    concat_txt = work_dir / "long_concat.txt"
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in segment_mp4s) + "\n", encoding="utf-8")
    raw_joined = work_dir / "long_joined.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(raw_joined)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    bgm_wav = work_dir / "ambient_bgm.wav"
    generate_ambient_bgm(cur_time, bgm_wav)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw_joined), "-i", str(bgm_wav),
        "-filter_complex", "[0:a]volume=1.0[vo];[1:a]volume=0.85[bg];[vo][bg]amix=inputs=2:duration=first:dropout_transition=2,loudnorm=I=-14.2:TP=-1.2:LRA=4.5[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-movflags", "+faststart",
        str(out_mp4),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return {"duration": round(cur_time, 2), "chapters": chapter_timestamps, "path": str(out_mp4)}


def assemble_short_video(
    plan: dict,
    tts_dir: Path,
    img_dir: Path,
    thumb_path: Path,
    out_mp4: Path,
    work_dir: Path,
    fps: int = 30,
) -> dict:
    """Render 1080x1920@30fps Traffic-Funnel Short synced to Alan's cloned voiceover."""
    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = work_dir / "short_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    tts_meta = json.loads((tts_dir / "tts_results.json").read_text(encoding="utf-8"))
    dur_map = {it["id"]: float(it["duration"]) for it in tts_meta["items"]}

    thumb_img = Image.open(thumb_path).convert("RGB") if thumb_path.exists() else None
    sbeats = plan.get("short_beats", [])
    step_beats = [b for b in sbeats if b.get("visual_type") == "step_phone_mockup"]
    step_total = max(1, len(step_beats))

    segment_mp4s = []
    cur_step = 0
    total_dur = 0.0

    for idx, beat in enumerate(sbeats):
        bid = beat["id"]
        wav_path = tts_dir / f"{bid}.wav"
        dur = dur_map.get(bid, 7.0)
        n_frames = max(fps, int(round(dur * fps)))
        real_dur = n_frames / fps

        if beat.get("visual_type") == "step_phone_mockup":
            cur_step += 1

        bg_path = img_dir / f"{bid}.png"
        if not bg_path.exists():
            cands = sorted(list(img_dir.glob("S*.png")) + list(img_dir.glob("L*.png")))
            bg_path = cands[idx % len(cands)] if cands else bg_path
        bg_img = Image.open(bg_path).convert("RGB") if bg_path.exists() else None

        seg_mp4 = clips_dir / f"{bid}.mp4"
        ff_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1080x1920", "-r", str(fps), "-i", "-",
            "-i", str(wav_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-t", f"{real_dur:.3f}",
            str(seg_mp4),
        ]
        proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None

        for fi in range(n_frames):
            t_loc = fi / fps
            prog = fi / max(1, n_frames - 1)
            cap = _caption_at_time(beat.get("captions", []), t_loc, real_dur)
            frame = render_short_frame(
                beat=beat,
                bg_img=bg_img,
                thumb_img=thumb_img,
                caption_text=cap,
                step_idx=cur_step,
                step_total=step_total,
                beat_progress=prog,
            )
            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.wait()
        segment_mp4s.append(seg_mp4)
        total_dur += real_dur
        print(f"  [Compositor:Short] Rendered {bid} ({beat.get('visual_type')}, {real_dur:.2f}s)")

    concat_txt = work_dir / "short_concat.txt"
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in segment_mp4s) + "\n", encoding="utf-8")
    raw_joined = work_dir / "short_joined.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(raw_joined)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    bgm_wav = work_dir / "short_bgm.wav"
    generate_ambient_bgm(total_dur, bgm_wav)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(raw_joined), "-i", str(bgm_wav),
        "-filter_complex", "[0:a]volume=1.0[vo];[1:a]volume=0.70[bg];[vo][bg]amix=inputs=2:duration=first:dropout_transition=2,loudnorm=I=-14.5:TP=-1.2:LRA=4.0[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-movflags", "+faststart",
        str(out_mp4),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return {"duration": round(total_dur, 2), "path": str(out_mp4)}
