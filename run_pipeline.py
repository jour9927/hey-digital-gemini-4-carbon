#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon — Unattended End-to-End Video & Shorts Production Agent
====================================================================================
Master Orchestrator (ALWAYS runs and stores master state locally on `Alan-Hub` first!):

1. [Cluster Check] Probes Colab RTX Pro 6000 HTTPS Bridge + Tailscale SSH nodes (`b850`, `alan-fw`, `alan-mba`).
2. [Crawler & Topic Agent] Unattended topic discovery from live public tech feeds (or `--topic` override)
   + crawls multi-source public specs, benchmarks, and news (`[SRC-01]..[SRC-12]`).
3. [AI Script & Storyboard Agent] Generates Hey Digital Long-Form script + corresponding Traffic-Funnel
   Short script + visual beat plans + cited YouTube descriptions.
4. [Heavy GPU Stage on RTX Pro 6000] Uploads master worker script & Alan's reference voice to
   `/content/heydigital_g4c`, runs PyTorch Demucs/Whisper voice profiling, PyTorch F5-TTS voice cloning,
   and AI hardware studio image synthesis, pulling all generated WAVs & PNGs immediately back to local disk!
5. [Local Master Compositor] Renders `1920x1080@30fps` Long-Form MP4, `1080x1920@30fps` Traffic-Funnel
   Short MP4, `1920x1080` Spotlight Thumbnail PNG, and full Cited Metadata Markdown/JSON!
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image

from hey_digital_pipeline.compositor import (
    assemble_long_video,
    assemble_short_video,
    render_thumbnail,
)
from hey_digital_pipeline.crawler import crawl_topic_dossier, discover_unattended_topics
from hey_digital_pipeline.remote_worker import ColabBridgeConfig, ColabRTX6000Client, TailscaleClusterInspector
from hey_digital_pipeline.scriptwriter import formulate_topic_from_candidates, generate_production_plan

REPO_ROOT = Path(__file__).resolve().parent


def load_existing_channel_titles() -> list[str]:
    cat = REPO_ROOT / "config" / "channel_history.json"
    if cat.exists():
        data = json.loads(cat.read_text(encoding="utf-8"))
        return [x.get("title", "") for x in data if x.get("title")]
    return []


def format_youtube_descriptions(plan: dict, long_chapters: list[dict]) -> tuple[str, str]:
    ch_lines = "\n".join(f"{c['time']} {c['title']}" for c in long_chapters)
    src_lines = "\n".join(
        f"[{s['source_id']}] {s['publisher']} — {s['title']}\n    {s['url']}"
        for s in plan.get("cited_sources", [])
    )
    desc_long = (
        f"{plan.get('summary_paragraph', '')}\n\n"
        f"⏱️ 時間軸章節\n{ch_lines}\n\n"
        f"📚 本期公開爬蟲資訊與實測數據來源\n{src_lines}\n\n"
        f"🔔 喜歡這類硬核不灌水的科技選購與實測拆解，記得訂閱 Hey Digital 並開啟小鈴铛！\n"
        f"#HeyDigital #科技評測 #選購指南 #AI硬體"
    )
    short_src_lines = "\n".join(
        f"[{s['source_id']}] {s['publisher']} — {s['title']}\n    {s['url']}"
        for s in plan.get("cited_sources", [])[:4]
    )
    desc_short = (
        f"{plan.get('summary_paragraph', '')[:160]}\n\n"
        f"👉 完整多維度實測對比與公開數據來源，請點擊下方關聯長片：\n"
        f"《{plan.get('title_long', '')}》\n\n"
        f"📚 核心參考來源：\n{short_src_lines}\n\n"
        f"#Shorts #HeyDigital #科技避坑"
    )
    return desc_long, desc_short


def ensure_remote_deps(client: ColabRTX6000Client):
    """Bootstrap isolated venv/packages on RTX Pro 6000 if not already installed."""
    boot_cmd = (
        "set -e; "
        "python3 -c 'import f5_tts, faster_whisper, diffusers, transformers, accelerate, soundfile' 2>/dev/null || "
        "pip install -q f5-tts faster-whisper diffusers transformers accelerate safetensors soundfile scipy demucs"
    )
    jid = client.exec_async(boot_cmd, cwd="repo", job_id="bootstrap_deps")
    client.wait_job(jid, poll_sec=4.0, max_wait_sec=900.0)


def run_pipeline(args: argparse.Namespace) -> Path:
    slug_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / (args.run_id or f"ep_{slug_ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Hey Digital Gemini 4 Carbon Pipeline | Run: {run_dir.name} ===")

    # Step 1: Probe Tailscale SSH Cluster + RTX Pro 6000 Bridge
    inspector = TailscaleClusterInspector(REPO_ROOT)
    cluster_status = inspector.probe_all()
    (run_dir / "cluster_status.json").write_text(json.dumps(cluster_status, ensure_ascii=False, indent=2), encoding="utf-8")
    for n in cluster_status:
        print(f"  [TailscaleNode] {n['host']}: reachable={n['reachable']} | {n.get('hostname','')} | gpu={n.get('gpu','')}")

    bridge_cfg_file = REPO_ROOT / "config" / "rtx6000_bridge.json"
    bridge_url = args.colab_url or (json.loads(bridge_cfg_file.read_text())["url"] if bridge_cfg_file.exists() else "")
    bridge_tok = args.colab_token or (json.loads(bridge_cfg_file.read_text()).get("token", "hd_g4c_rtx6000_9f8a7b6c5d4e3f2a1b0c9d8e") if bridge_cfg_file.exists() else "hd_g4c_rtx6000_9f8a7b6c5d4e3f2a1b0c9d8e")
    if not bridge_url:
        raise RuntimeError("Colab RTX Pro 6000 bridge URL required (--colab-url or config/rtx6000_bridge.json)")

    client = ColabRTX6000Client(ColabBridgeConfig(url=bridge_url, token=bridge_tok))
    h = client.health()
    print(f"  [RTX6000Bridge] Connected! GPU={h.get('gpu')} | CPUs={h.get('cpus')} | FreeDisk={h.get('disk_free_gb')}GB")

    # Step 2: Unattended Topic Discovery & Public Web Crawling
    plan_file = run_dir / "production_plan.json"
    dossier_file = run_dir / "crawled_dossier.json"
    existing_titles = load_existing_channel_titles()

    if plan_file.exists() and not args.force_regen_script:
        print("[Stage 2-3] Reusing existing local master production_plan.json")
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    else:
        if args.topic:
            selected_topic = args.topic
            queries = args.queries.split(",") if args.queries else [selected_topic, "Apple M4 Pro Mac mini memory bandwidth", "AMD Ryzen AI Max 395 Local LLM"]
            wiki_terms = queries[:2]
        else:
            print("[Stage 2A] Unattended AI Agent scanning live public tech RSS & Hacker News feeds...")
            cand_report = discover_unattended_topics(existing_titles, top_k=6)
            (run_dir / "topic_candidates.json").write_text(json.dumps(cand_report, ensure_ascii=False, indent=2), encoding="utf-8")
            framed = formulate_topic_from_candidates(cand_report, existing_titles, video_type=args.video_type)
            (run_dir / "selected_topic.json").write_text(json.dumps(framed, ensure_ascii=False, indent=2), encoding="utf-8")
            selected_topic = framed["selected_topic"]
            queries = framed.get("search_queries") or [selected_topic]
            wiki_terms = framed.get("wiki_terms") or queries[:2]
            print(f"  -> Unattended Agent Selected Topic: {selected_topic}")

        print(f"[Stage 2B] Crawling public web sources & specs for: {selected_topic} ...")
        dossier = crawl_topic_dossier(selected_topic, search_queries=queries, wiki_terms=wiki_terms)
        dossier_file.write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> Crawled {dossier['source_count']} verifiable public sources ([SRC-01]..[SRC-{dossier['source_count']:02d}])")

        print("[Stage 3] Generating cited Long-Form + Traffic-Funnel Short script & storyboard...")
        plan = generate_production_plan(
            topic=selected_topic,
            video_type=args.video_type,
            dossier=dossier,
            target_long_beats=args.long_beats,
            target_short_steps=args.short_steps,
        )
        plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> Script ready! Long title: {plan.get('title_long')} ({len(plan.get('long_beats', []))} beats, Short: {len(plan.get('short_beats', []))} beats)")

    # Step 4: Heavy PyTorch Voice Cloning & AI Image Synthesis on RTX Pro 6000
    print("[Stage 4] Syncing master heavy-worker script to RTX Pro 6000 & checking dependencies...")
    client.put_file(REPO_ROOT / "worker" / "rtx6000_heavy_worker.py", "repo/rtx6000_heavy_worker.py")
    ensure_remote_deps(client)

    # 4A: Voice Profile on RTX Pro 6000 (cached across runs!)
    local_vp_dir = REPO_ROOT / "assets" / "voices" / "rtx6000_profile"
    if not (local_vp_dir / "voice_profile.json").exists():
        print("[Stage 4A] Extracting pure studio vocal profile of Alan's voice on RTX Pro 6000...")
        ref_src = REPO_ROOT / "assets" / "voices" / "alan_heydigital_ref_24s.wav"
        client.put_file(ref_src, "voices/alan_heydigital_ref_24s.wav")
        jid_vp = client.exec_async(
            "python3 -u rtx6000_heavy_worker.py voice-profile --input-wav ../voices/alan_heydigital_ref_24s.wav --out-dir ../voices/profile",
            cwd="repo",
            job_id=f"vp_{slug_ts}",
        )
        client.wait_job(jid_vp, poll_sec=3.0, max_wait_sec=600.0)
        local_vp_dir.mkdir(parents=True, exist_ok=True)
        client.get_file("voices/profile/voice_profile.json", local_vp_dir / "voice_profile.json")
        client.get_file("voices/profile/alan_prompt_clip_24k.wav", local_vp_dir / "alan_prompt_clip_24k.wav")
    else:
        # Ensure remote has profile if Colab restarted
        client.put_file(local_vp_dir / "alan_prompt_clip_24k.wav", "voices/profile/alan_prompt_clip_24k.wav")
        vp_obj = json.loads((local_vp_dir / "voice_profile.json").read_text(encoding="utf-8"))
        vp_obj["ref_wav"] = "/content/heydigital_g4c/voices/profile/alan_prompt_clip_24k.wav"
        client.put_bytes(json.dumps(vp_obj, ensure_ascii=False, indent=2).encode("utf-8"), "voices/profile/voice_profile.json")

    # 4B: Batch PyTorch F5-TTS Voice Cloning for Long + Short on RTX Pro 6000
    tts_local_dir = run_dir / "tts"
    if not (tts_local_dir / "tts_results.json").exists():
        tts_items = [{"id": b["id"], "text": b["narration"]} for b in plan.get("long_beats", [])] + [
            {"id": sb["id"], "text": sb["narration"]} for sb in plan.get("short_beats", [])
        ]
        tts_manifest = {"speed": 1.03, "items": tts_items}
        (run_dir / "tts_manifest.json").write_text(json.dumps(tts_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        client.put_file(run_dir / "tts_manifest.json", f"jobs/{run_dir.name}/tts_manifest.json")

        print(f"[Stage 4B] Synthesizing {len(tts_items)} cloned voiceover segments on RTX Pro 6000...")
        jid_tts = client.exec_async(
            f"python3 -u rtx6000_heavy_worker.py tts-batch --manifest ../jobs/{run_dir.name}/tts_manifest.json --profile ../voices/profile/voice_profile.json --out-dir ../jobs/{run_dir.name}/tts",
            cwd="repo",
            job_id=f"tts_{run_dir.name}",
        )
        client.wait_job(jid_tts, poll_sec=4.0, max_wait_sec=1800.0)
        client.get_tar_dir(f"jobs/{run_dir.name}/tts", tts_local_dir)
        print(f"  -> Downloaded all {len(tts_items)} cloned WAVs to local {tts_local_dir}!")

    # 4C: Batch AI Studio Image Generation on RTX Pro 6000
    img_local_dir = run_dir / "images"
    if not (img_local_dir / "image_results.json").exists():
        img_items = []
        th_prompt = (plan.get("thumbnail") or {}).get("image_prompt") or "Minimalist dark studio spotlight product shot of sleek aluminum Mac computer and AI chip, rim lighting, matte black background, no text, 8k"
        img_items.append({"id": "THUMB_HERO", "prompt": th_prompt, "width": 1344, "height": 768, "seed": 8800})

        for idx, b in enumerate(plan.get("long_beats", [])):
            if b.get("visual_type") in ("cinema_tech_broll", "chapter_dark_card") and b.get("image_prompt"):
                img_items.append({
                    "id": b["id"],
                    "prompt": b["image_prompt"],
                    "width": 1280,
                    "height": 720,
                    "seed": 5100 + idx,
                })
        for idx, sb in enumerate(plan.get("short_beats", [])):
            if sb.get("visual_type") != "cta_funnel_endcard":
                sp = sb.get("image_prompt") or th_prompt
                img_items.append({
                    "id": sb["id"],
                    "prompt": sp,
                    "width": 720,
                    "height": 1280 if sb.get("visual_type") == "hook_sticker_broll" else 720,
                    "seed": 6100 + idx,
                })

        img_manifest = {"items": img_items}
        (run_dir / "img_manifest.json").write_text(json.dumps(img_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        client.put_file(run_dir / "img_manifest.json", f"jobs/{run_dir.name}/img_manifest.json")

        print(f"[Stage 4C] Generating {len(img_items)} photorealistic studio visuals on RTX Pro 6000...")
        jid_img = client.exec_async(
            f"python3 -u rtx6000_heavy_worker.py img-batch --manifest ../jobs/{run_dir.name}/img_manifest.json --out-dir ../jobs/{run_dir.name}/images",
            cwd="repo",
            job_id=f"img_{run_dir.name}",
        )
        client.wait_job(jid_img, poll_sec=4.0, max_wait_sec=1800.0)
        client.get_tar_dir(f"jobs/{run_dir.name}/images", img_local_dir)
        print(f"  -> Downloaded all {len(img_items)} generated studio visuals to local {img_local_dir}!")

    # Step 5: Local Master Rendering (Thumbnail + Long-Form + Traffic-Funnel Short + Cited Metadata)
    deliverables_dir = run_dir / "deliverables"
    deliverables_dir.mkdir(parents=True, exist_ok=True)

    hero_png = img_local_dir / "THUMB_HERO.png"
    hero_im = Image.open(hero_png).convert("RGB") if hero_png.exists() else None
    thumb_out = deliverables_dir / "thumbnail_1080p.png"
    render_thumbnail(plan, hero_im, thumb_out)
    print(f"[Stage 5A] Thumbnail rendered: {thumb_out}")

    long_out = deliverables_dir / "hey_digital_long_1080p.mp4"
    long_meta_cache = run_dir / "long_render_info.json"
    if long_out.exists() and long_meta_cache.exists() and not args.force_regen_script:
        long_info = json.loads(long_meta_cache.read_text(encoding="utf-8"))
        print(f"[Stage 5B] Reusing rendered Long-Form Video: {long_out} ({long_info['duration']}s)")
    else:
        long_info = assemble_long_video(plan, tts_local_dir, img_local_dir, long_out, run_dir / "work_long")
        long_meta_cache.write_text(json.dumps(long_info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Stage 5B] Long-Form Video rendered: {long_out} ({long_info['duration']}s)")

    short_out = deliverables_dir / "hey_digital_short_funnel_1080x1920.mp4"
    short_info = assemble_short_video(plan, tts_local_dir, img_local_dir, thumb_out, short_out, run_dir / "work_short")
    print(f"[Stage 5C] Traffic-Funnel Short rendered: {short_out} ({short_info['duration']}s)")

    desc_long, desc_short = format_youtube_descriptions(plan, long_info["chapters"])
    (deliverables_dir / "youtube_long_metadata.txt").write_text(
        f"TITLE:\n{plan.get('title_long', '')}\n\nDESCRIPTION:\n{desc_long}\n", encoding="utf-8"
    )
    (deliverables_dir / "youtube_short_metadata.txt").write_text(
        f"TITLE:\n{plan.get('title_short', '')}\n\nRELATED LONG VIDEO:\n{plan.get('title_long', '')}\n\nDESCRIPTION:\n{desc_short}\n", encoding="utf-8"
    )
    summary = {
        "run_id": run_dir.name,
        "topic": plan.get("topic"),
        "video_type": plan.get("video_type"),
        "title_long": plan.get("title_long"),
        "title_short": plan.get("title_short"),
        "long_video": str(long_out),
        "long_duration_sec": long_info["duration"],
        "short_video": str(short_out),
        "short_duration_sec": short_info["duration"],
        "thumbnail": str(thumb_out),
        "cited_sources_count": len(plan.get("cited_sources", [])),
        "cited_sources": plan.get("cited_sources", []),
    }
    (deliverables_dir / "release_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== PIPELINE COMPLETE! Deliverables saved locally to: {deliverables_dir} ===")
    return deliverables_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Hey Digital Gemini 4 Carbon — Autonomous Video & Shorts Pipeline")
    ap.add_argument("--topic", default="", help="Optional manual topic override; leave empty for 100%% unattended topic discovery")
    ap.add_argument("--queries", default="", help="Comma-separated search queries when --topic is provided")
    ap.add_argument("--video-type", default="versus_guide", choices=["versus_guide", "ai_benchmark", "deep_review", "hidden_tips", "news_analysis"])
    ap.add_argument("--long-beats", type=int, default=20, help="Target number of narrative/visual beats for Long-form video")
    ap.add_argument("--short-steps", type=int, default=5, help="Target number of structured steps in Traffic-Funnel Short")
    ap.add_argument("--colab-url", default="", help="HTTPS trycloudflare URL for RTX Pro 6000 bridge")
    ap.add_argument("--colab-token", default="", help="Bearer token for RTX Pro 6000 bridge")
    ap.add_argument("--run-id", default="", help="Custom run folder name under runs/")
    ap.add_argument("--force-regen-script", action="store_true")
    return ap


if __name__ == "__main__":
    run_pipeline(build_arg_parser().parse_args())
