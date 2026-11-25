#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon V4 — Automatic Telegram Deliverable Push
====================================================================
Automatically sends the rendered Spotlight Thumbnail (`1920x1080`), Long-Form Video (`1920x1080`),
Interactive Traffic-Funnel Short (`1080x1920`), and Standalone Voiceover Master MP3 directly to the
channel owner's Telegram chat (`supports_streaming=true`), dynamically capping bitrate for >45MB
MP4s and generating spec-compliant `320x180` JPEG video previews.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def resolve_telegram_credentials() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("OPENCLAW_TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""

    home = Path.home()
    for profile_dir in (home / ".openclaw-realtime", home / ".openclaw"):
        env_f = profile_dir / "service-env" / "ai.openclaw.realtime.env"
        if not token and env_f.exists():
            for ln in env_f.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "TELEGRAM" in ln and "=" in ln and not ln.strip().startswith("#"):
                    val = ln.split("=", 1)[1].strip().strip("\"'")
                    if ":" in val and not val.startswith("$"):
                        token = val
                        break
        cfg_f = profile_dir / "openclaw.json"
        if cfg_f.exists():
            try:
                cfg = json.loads(cfg_f.read_text(encoding="utf-8"))
                tg = (cfg.get("channels") or {}).get("telegram") or {}
                if not token:
                    bt = str(tg.get("botToken") or "").strip()
                    if ":" in bt and not bt.startswith("$"):
                        token = bt
                if not token and tg.get("tokenFile"):
                    tf = Path(str(tg["tokenFile"])).expanduser()
                    if tf.exists():
                        cand = tf.read_text(encoding="utf-8", errors="ignore").strip()
                        if ":" in cand:
                            token = cand
                if not chat_id:
                    af = tg.get("allowFrom") or []
                    if af:
                        chat_id = str(af[0]).replace("tg:", "").strip()
            except Exception:
                pass

    if not token or not chat_id:
        raise RuntimeError("Telegram credentials not found (set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID or configure ~/.openclaw-realtime).")
    return token, chat_id


def _tg_post(token: str, method: str, form_args: list[str], retries: int = 3) -> dict:
    last_err = ""
    for attempt in range(retries):
        proc = subprocess.run(
            ["curl", "-fsS", "--http1.1", "--max-time", "300", f"https://api.telegram.org/bot{token}/{method}"] + form_args,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            res = json.loads(proc.stdout)
            if res.get("ok"):
                return res["result"]
            last_err = f"API response not ok: {res.get('description')}"
        else:
            last_err = f"curl exit={proc.returncode}"
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Telegram API {method} failed after {retries} attempts ({last_err})")


def _probe_duration(mp4_path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(mp4_path)],
            text=True,
            timeout=10,
        ).strip()
        return max(10.0, float(out))
    except Exception:
        return 300.0


def ensure_tg_size_limit(mp4_path: Path, max_mb: float = 44.5) -> Path:
    max_bytes = int(max_mb * 1024 * 1024)
    if mp4_path.stat().st_size <= max_bytes:
        return mp4_path
    tg_mp4 = mp4_path.with_name(mp4_path.stem + "_tg.mp4")
    if tg_mp4.exists() and tg_mp4.stat().st_size <= max_bytes and tg_mp4.stat().st_mtime >= mp4_path.stat().st_mtime:
        return tg_mp4

    dur = _probe_duration(mp4_path)
    target_v_kbps = max(550, min(1850, int((max_mb * 0.94 * 8192.0) / dur) - 196))
    buf_kbps = target_v_kbps * 2

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(mp4_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-maxrate", f"{target_v_kbps}k", "-bufsize", f"{buf_kbps}k",
            "-c:a", "copy", "-movflags", "+faststart",
            str(tg_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tg_mp4


def _make_tg_video_thumb_jpg(thumb_png: Path) -> Path:
    """Telegram `sendVideo` requires a <=320px JPEG <200kB for inline streaming preview."""
    jpg_p = thumb_png.with_name("_tg_thumb_320.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(thumb_png), "-vf", "scale=320:180", "-q:v", "5", str(jpg_p)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return jpg_p


def _extract_tg_voice_mp3(long_mp4: Path) -> Path:
    mp3_p = long_mp4.with_name("hey_digital_v4_voice_master.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(long_mp4), "-vn", "-c:a", "libmp3lame", "-b:a", "192k", str(mp3_p)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return mp3_p


def send_run_to_telegram(deliverables_dir: Path) -> dict:
    token, chat_id = resolve_telegram_credentials()
    manifest = json.loads((deliverables_dir / "release_manifest.json").read_text(encoding="utf-8"))
    thumb_png = deliverables_dir / "thumbnail_1080p.png"
    thumb_320_jpg = _make_tg_video_thumb_jpg(thumb_png)
    master_long_mp4 = deliverables_dir / "hey_digital_long_1080p.mp4"
    long_mp4 = ensure_tg_size_limit(master_long_mp4)
    short_mp4 = ensure_tg_size_limit(deliverables_dir / "hey_digital_short_funnel_1080x1920.mp4")
    voice_mp3 = _extract_tg_voice_mp3(master_long_mp4)

    title_long = manifest.get("title_long", "")
    title_short = manifest.get("title_short", "")
    dur_long = int(round(float(manifest.get("long_duration_sec") or 309)))
    dur_short = int(round(float(manifest.get("short_duration_sec") or 91)))
    src_cnt = manifest.get("cited_sources_count", 0)

    r_photo = _tg_post(token, "sendPhoto", [
        "-F", f"chat_id={chat_id}",
        "-F", f"photo=@{thumb_png}",
        "-F", (
            f"caption=🎬 Hey Digital V5 字幕斷字＋字級對齊精修成片（{manifest.get('run_id')}）\n\n"
            f"1️⃣ 【1080p 頻道聚光燈縮圖】\n"
            f"📌 長片標題：《{title_long}》\n"
            f"📚 公開爬蟲與純硬體對位來源：{src_cnt} 筆"
        ),
    ])

    r_audio = _tg_post(token, "sendAudio", [
        "-F", f"chat_id={chat_id}",
        "-F", f"audio=@{voice_mp3}",
        "-F", f"thumbnail=@{thumb_320_jpg}",
        "-F", f"duration={dur_long}",
        "-F", "performer=Hey Digital (Alan Cloned V4)",
        "-F", f"title={title_long[:48]}",
        "-F", "caption=🎧 【V4 長片獨立母帶音軌 MP3｜可直接點開試聽咬字、破音字與呼吸感】",
    ])

    r_long = _tg_post(token, "sendVideo", [
        "-F", f"chat_id={chat_id}",
        "-F", f"video=@{long_mp4}",
        "-F", f"thumbnail=@{thumb_320_jpg}",
        "-F", "width=1920",
        "-F", "height=1080",
        "-F", f"duration={dur_long}",
        "-F", "supports_streaming=true",
        "-F", (
            f"caption=2️⃣ 【V5 長片成片 1920x1080@30fps｜{dur_long}s｜-14.2 LUFS】\n"
            f"• 中文詞法語法斷字引擎（視覺字寬單行排版、零破詞、零句尾懸掛虛詞、保留頓號並列與轉折詞）+ Whisper 字級時間軸同步\n"
            f"• KeypointRCNN 零人臉過濾 + SigLIP 逐句對位實拍 + Three.js 3D 模板 + 原聲級配音"
        ),
    ])

    r_short = _tg_post(token, "sendVideo", [
        "-F", f"chat_id={chat_id}",
        "-F", f"video=@{short_mp4}",
        "-F", "width=1080",
        "-F", "height=1920",
        "-F", f"duration={dur_short}",
        "-F", "supports_streaming=true",
        "-F", (
            f"caption=3️⃣ 【V4 互動導流 Short 1080x1920@30fps｜{dur_short}s｜-14.3 LUFS】\n"
            f"📌 標題：{title_short}"
        ),
    ])

    return {
        "chat_id": chat_id,
        "photo_message_id": r_photo.get("message_id"),
        "audio_message_id": r_audio.get("message_id"),
        "long_video_message_id": r_long.get("message_id"),
        "short_video_message_id": r_short.get("message_id"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    deliv = repo_root / "runs" / args.run_id / "deliverables"
    info = send_run_to_telegram(deliv)
    print("TELEGRAM_DELIVERED:", json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
