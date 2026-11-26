#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon V2 — Flagship Unattended End-to-End Video & Shorts Production Agent
================================================================================================
Master Orchestrator (ALWAYS runs and stores master state locally on `Alan-Hub` first!):

1. [Cluster & Ephemeral RAM-Disk Check]
   Probes Colab RTX Pro 6000 HTTPS Bridge + Tailscale SSH nodes (`b850`, `alan-fw`, `alan-mba`),
   and mounts `/dev/shm` high-speed workspace on RTX Pro 6000 so concurrent sessions never run low on disk.
2. [Deep Full-Text Crawler & Real B-Roll Harvester]
   Unattended topic discovery -> full-text MediaWiki + 9to5Mac/HN/RSS scraping (`[SRC-01]..[SRC-14]`)
   + `yt-dlp` real 1080p hardware showcase B-roll harvesting (`BROLL_01..BROLL_08`).
3. [Flagship 32B AI Script & High-Density 3D Storyboard Director]
   Runs `Qwen2.5-32B-Instruct` on RTX Pro 6000 (`:19465`) with verbatim `@hey_digitall` prose anchors
   and `cn_verbalizer` 100% Mandarin spoken normalization (`tts_text`).
4. [Heavy PyTorch GPU Stage on RTX Pro 6000]
   Releases `:19465` VRAM -> runs sentence-level PyTorch `F5-TTS` voice cloning + PyTorch
   `whisper-large-v3-turbo` QA & forced caption alignment + `RealVisXL_V4.0` studio stills +
   `Lightricks/LTX-Video` CUDA Image-to-Video (`I2V`) motion clips!
   Pulls ALL WAVs, PNGs, MP4s, and JSON manifests back to local `Alan-Hub` disk immediately!
5. [Local Metal WebGL2 `Three.js` + Rhodes Sidechain Compositor]
   Renders `1920x1080@30fps` Long-Form MP4, `1080x1920@30fps` Interactive Traffic-Funnel Short MP4,
   `1920x1080` Spotlight Thumbnail PNG, and full Cited YouTube Metadata!
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from PIL import Image

from hey_digital_pipeline.broll_matcher import match_broll_slots_semantically
from hey_digital_pipeline.compositor import (
    assemble_long_video,
    assemble_short_video,
    render_thumbnail,
)
from hey_digital_pipeline.crawler import (
    crawl_topic_dossier,
    discover_unattended_topics,
)
from hey_digital_pipeline.remote_worker import (
    ColabBridgeConfig,
    ColabRTX6000Client,
    TailscaleClusterInspector,
)
from hey_digital_pipeline.scriptwriter import (
    formulate_topic_from_candidates,
    generate_production_plan,
)

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
        f"📚 本期公開爬蟲資訊、實測數據與畫面參考來源\n{src_lines}\n\n"
        f"🔔 喜歡這類硬核不灌水的科技選購與底層刀法拆解，記得訂閱 Hey Digital 並開啟小鈴鐺！\n"
        f"#HeyDigital #科技評測 #選購指南 #Apple #AI硬體"
    )
    short_src_lines = "\n".join(
        f"[{s['source_id']}] {s['publisher']} — {s['title']}\n    {s['url']}"
        for s in plan.get("cited_sources", [])[:5]
    )
    desc_short = (
        f"{plan.get('summary_paragraph', '')[:160]}\n\n"
        f"👉 完整多維度實測對比、3D 頻寬拆解與公開數據來源，請點擊下方關聯長片：\n"
        f"《{plan.get('title_long', '')}》\n\n"
        f"📚 核心參考來源：\n{short_src_lines}\n\n"
        f"#Shorts #HeyDigital #科技避坑 #選購指南"
    )
    return desc_long, desc_short


def prepare_remote_workspace_and_deps(client: ColabRTX6000Client):
    """Ensure isolated remote workspace & PyTorch/F5-TTS/SigLIP/Diffusers dependencies on RTX Pro 6000."""
    cmd = (
        "set -e; "
        "mkdir -p ../jobs ../hf_cache/hub /dev/shm/heydigital_hf/hub; "
        "for m in ../hf_cache/hub/models--*; do "
        "  [ -e \"$m\" ] && ln -sfn \"$(realpath \"$m\")\" /dev/shm/heydigital_hf/hub/$(basename \"$m\"); "
        "done; "
        "python3 -c 'import f5_tts, diffusers, transformers, accelerate, soundfile, sentencepiece, bitsandbytes' 2>/dev/null || "
        "pip install -q f5-tts diffusers transformers accelerate safetensors soundfile scipy demucs sentencepiece protobuf bitsandbytes imageio imageio-ffmpeg"
    )
    jid = client.exec_async(cmd, cwd="repo", job_id="prep_v3_deps")
    client.wait_job(jid, poll_sec=3.0, max_wait_sec=900.0)


def ensure_remote_32b_llm(client: ColabRTX6000Client):
    """Ensure RTX Pro 6000 Qwen2.5 server (`:19465`) is online for Stage 2-3 (uses PID file to avoid pgrep self-match)."""
    client.put_file(REPO_ROOT / "worker" / "rtx6000_llm_server.py", "repo/rtx6000_llm_server.py")
    cmd = (
        "mkdir -p ../logs; "
        "if curl -fsS http://127.0.0.1:19465/health 2>/dev/null; then exit 0; fi; "
        "if ! ([ -f ../logs/llm_19465.pid ] && kill -0 $(cat ../logs/llm_19465.pid) 2>/dev/null); then "
        "  nohup python3 -u rtx6000_llm_server.py --port 19465 > ../logs/llm_19465.log 2>&1 & "
        "  echo $! > ../logs/llm_19465.pid; "
        "fi; "
        "for i in $(seq 1 240); do "
        "  if curl -fsS http://127.0.0.1:19465/health 2>/dev/null; then echo; exit 0; fi; "
        "  sleep 2; "
        "done; tail -n 30 ../logs/llm_19465.log 2>/dev/null || true; exit 1"
    )
    jid = client.exec_async(cmd, cwd="repo", job_id="ensure_32b")
    client.wait_job(jid, poll_sec=3.0, max_wait_sec=520.0)


def release_remote_32b_llm(client: ColabRTX6000Client):
    """Free the ~61 GB of 32B LLM VRAM right before Stage 4 so F5-TTS + Whisper + RealVisXL + LTX-Video run at max speed."""
    jid = client.exec_async("fuser -k 19465/tcp 2>/dev/null || true; sleep 2; nvidia-smi --query-gpu=memory.free --format=csv,noheader", cwd="repo", job_id="free_32b_vram")
    client.wait_job(jid, poll_sec=1.5, max_wait_sec=30.0)


def run_pipeline(args: argparse.Namespace) -> Path:
    slug_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "runs" / (args.run_id or f"ep_v2_{slug_ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Hey Digital Gemini 4 Carbon V2 Flagship Pipeline | Run: {run_dir.name} ===")

    # Step 1: Probe Tailscale SSH Cluster + RTX Pro 6000 Bridge
    inspector = TailscaleClusterInspector(REPO_ROOT)
    cluster_status = inspector.probe_all()
    (run_dir / "cluster_status.json").write_text(json.dumps(cluster_status, ensure_ascii=False, indent=2), encoding="utf-8")
    for n in cluster_status:
        print(f"  [TailscaleNode] {n['host']}: reachable={n['reachable']} | {n.get('hostname','')} | gpu={n.get('gpu','')}")

    bridge_cfg_file = REPO_ROOT / "config" / "rtx6000_bridge.json"
    bridge_url = args.colab_url or (json.loads(bridge_cfg_file.read_text())["url"] if bridge_cfg_file.exists() else "")
    bridge_tok = args.colab_token or (json.loads(bridge_cfg_file.read_text()).get("token", "") if bridge_cfg_file.exists() else "")

    client: ColabRTX6000Client | None = None
    if bridge_url:
        try:
            cand_client = ColabRTX6000Client(ColabBridgeConfig(url=bridge_url, token=bridge_tok))
            h = cand_client.health()
            client = cand_client
            print(f"  [RTX6000Bridge] Connected! GPU={h.get('gpu')} | CPUs={h.get('cpus')}")
            prepare_remote_workspace_and_deps(client)
        except Exception as exc:
            print(f"  [RTX6000Bridge] Ephemeral tunnel offline ({exc}); seamlessly running on Local Apple Metal M3 Pro + cached master assets!")

    # Step 2 & 3: Unattended Topic Discovery, Deep Full-Text Crawler, Real B-Roll Harvester & 32B Scriptwriting
    plan_file = run_dir / "production_plan.json"
    dossier_file = run_dir / "crawled_dossier.json"
    broll_dir = run_dir / "broll"
    existing_titles = load_existing_channel_titles()

    if plan_file.exists() and not args.force_regen_script:
        print("[Stage 2-3] Reusing existing local master production_plan.json")
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    else:
        if client is not None:
            ensure_remote_32b_llm(client)
        sel_file = run_dir / "selected_topic.json"
        if dossier_file.exists() and sel_file.exists() and not args.force_regen_script:
            framed = json.loads(sel_file.read_text(encoding="utf-8"))
            selected_topic = framed["selected_topic"]
            dossier = json.loads(dossier_file.read_text(encoding="utf-8"))
            print(f"[Stage 2] Reusing cached topic & {dossier['source_count']} crawled sources for: {selected_topic}")
        else:
            if args.topic:
                selected_topic = args.topic
                queries = args.queries.split(",") if args.queries else [
                    selected_topic,
                    "MacBook Air vs MacBook Pro memory bandwidth display review",
                    "Apple Silicon unified memory bandwidth SSD speed benchmark",
                ]
                wiki_terms = ["MacBook Pro", "MacBook Air", "Apple silicon"]
            else:
                print("[Stage 2A] Unattended AI Agent scanning live public tech RSS & Hacker News feeds...")
                cand_report = discover_unattended_topics(existing_titles, top_k=7)
                (run_dir / "topic_candidates.json").write_text(json.dumps(cand_report, ensure_ascii=False, indent=2), encoding="utf-8")
                framed = formulate_topic_from_candidates(cand_report, existing_titles, video_type=args.video_type)
                sel_file.write_text(json.dumps(framed, ensure_ascii=False, indent=2), encoding="utf-8")
                selected_topic = framed["selected_topic"]
                queries = framed.get("search_queries") or [selected_topic]
                wiki_terms = framed.get("wiki_terms") or queries[:2]
                print(f"  -> Unattended Agent Selected Topic: {selected_topic}")

            print(f"[Stage 2B] Deep Full-Text Crawling public specs & articles for: {selected_topic} ...")
            dossier = crawl_topic_dossier(selected_topic, search_queries=queries, wiki_terms=wiki_terms)
            dossier_file.write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  -> Crawled {dossier['source_count']} deep public sources | Metrics pool: {dossier.get('verified_metrics_pool', [])[:10]}")

        print("[Stage 3] Generating V3 High-Density Script + Sub-Clause B-Roll Queries & 3D Storyboard...")
        plan = generate_production_plan(
            topic=selected_topic,
            video_type=args.video_type,
            dossier=dossier,
            target_long_beats=args.long_beats,
            target_short_steps=args.short_steps,
            ckpt_dir=run_dir / "script_ckpts",
        )
        plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> V3 Script & Storyboard saved locally! Long: {len(plan['long_beats'])} beats | Short: {len(plan['short_beats'])} beats")

    # Free `:19465` 32B LLM VRAM before Stage 3B/4 heavy Vision, Voice & Image models
    remote_hf_env = {"HF_HOME": "/dev/shm/heydigital_hf"}
    if client is not None:
        release_remote_32b_llm(client)
        client.put_file(REPO_ROOT / "worker" / "rtx6000_heavy_worker.py", "repo/rtx6000_heavy_worker.py")

    # Step 3B: Multi-Query Third-Party Shot Segmentation + KeypointRCNN Zero-Face + SigLIP Phrase-Level Visual Matching!
    if not (broll_dir / "matched_assignments.json").exists() or args.force_regen_script:
        print("[Stage 3B] Harvesting multi-query third-party hardware footage & running Zero-Face + SigLIP phrase-level visual matching...")
        match_stats = match_broll_slots_semantically(
            plan=plan,
            broll_dir=broll_dir,
            remote_client=client,
            remote_hf_env=remote_hf_env,
            run_name=run_dir.name,
        )
        plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> Matched {match_stats['slots']} narration clauses across {match_stats['candidates']} clean shots ({match_stats.get('unique_used', 0)} unique clips)!")

    # Step 4A: Studio Voice Profile on RTX Pro 6000 (or local cached profile)
    local_vp_dir = REPO_ROOT / "assets" / "voices" / "rtx6000_profile"
    if not (local_vp_dir / "voice_profile.json").exists() and client is not None:
        print("[Stage 4A] Extracting pure studio vocal profile of Alan's voice on RTX Pro 6000...")
        ref_src = REPO_ROOT / "assets" / "voices" / "alan_heydigital_ref_24s.wav"
        client.put_file(ref_src, "voices/alan_heydigital_ref_24s.wav")
        jid_vp = client.exec_async(
            "python3 -u rtx6000_heavy_worker.py voice-profile --input-wav ../voices/alan_heydigital_ref_24s.wav --out-dir ../voices/profile",
            cwd="repo",
            env=remote_hf_env,
            job_id=f"vp_{slug_ts}",
        )
        client.wait_job(jid_vp, poll_sec=3.0, max_wait_sec=600.0)
        local_vp_dir.mkdir(parents=True, exist_ok=True)
        client.get_file("voices/profile/voice_profile.json", local_vp_dir / "voice_profile.json")
        client.get_file("voices/profile/alan_prompt_clip_24k.wav", local_vp_dir / "alan_prompt_clip_24k.wav")
    elif client is not None and (local_vp_dir / "voice_profile.json").exists():
        vp_obj = json.loads((local_vp_dir / "voice_profile.json").read_text(encoding="utf-8"))
        long_ref_local = REPO_ROOT / vp_obj["ref_wav"] if not Path(vp_obj["ref_wav"]).is_absolute() else Path(vp_obj["ref_wav"])
        if not long_ref_local.exists():
            long_ref_local = local_vp_dir / "alan_v6_long_m5_24k.wav"
        client.put_file(long_ref_local, f"voices/profile/{long_ref_local.name}")
        vp_obj["ref_wav"] = f"/content/heydigital_g4c/voices/profile/{long_ref_local.name}"

        if vp_obj.get("short_ref_wav"):
            short_ref_local = REPO_ROOT / vp_obj["short_ref_wav"] if not Path(vp_obj["short_ref_wav"]).is_absolute() else Path(vp_obj["short_ref_wav"])
            if short_ref_local.exists():
                client.put_file(short_ref_local, f"voices/profile/{short_ref_local.name}")
                vp_obj["short_ref_wav"] = f"/content/heydigital_g4c/voices/profile/{short_ref_local.name}"
        client.put_bytes(json.dumps(vp_obj, ensure_ascii=False, indent=2).encode("utf-8"), "voices/profile/voice_profile.json")

    # Step 4B: V6 Dual-Engine Voice Cloning (Qwen3-TTS-1.7B / F5-TTS V6) + Whisper Forced Alignment
    tts_local_dir = run_dir / "tts"
    if not (tts_local_dir / "tts_results.json").exists():
        from hey_digital_pipeline.cn_verbalizer import purify_tw_tech_text, verbalize_for_mandarin_tts
        from hey_digital_pipeline.subtitle_engine import segment_narration_to_captions

        tts_items = []
        for b in plan.get("long_beats", []):
            nar = purify_tw_tech_text(b.get("narration", "")) if b.get("narration") else ""
            b["narration"] = nar
            b["captions"] = segment_narration_to_captions(nar, is_vertical=False) if nar else []
            b["tts_text"] = verbalize_for_mandarin_tts(nar) if nar else ""
            tts_items.append({
                "id": b["id"],
                "text": nar,
                "tts_text": b["tts_text"],
                "captions": b["captions"],
                "silent_duration": b.get("silent_duration", 1.85),
            })
        for sb in plan.get("short_beats", []):
            nar = purify_tw_tech_text(sb.get("narration", "")) if sb.get("narration") else ""
            sb["narration"] = nar
            sb["captions"] = segment_narration_to_captions(nar, is_vertical=True) if nar else []
            sb["tts_text"] = verbalize_for_mandarin_tts(nar) if nar else ""
            tts_items.append({
                "id": sb["id"],
                "text": nar,
                "tts_text": sb["tts_text"],
                "captions": sb["captions"],
            })
        plan_file.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        tts_manifest = {"engine": "qwen3", "speed": 1.01, "short_speed": 1.03, "items": tts_items}
        (run_dir / "tts_manifest.json").write_text(json.dumps(tts_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        if client is not None:
            client.put_file(run_dir / "tts_manifest.json", f"jobs/{run_dir.name}/tts_manifest.json")
            print(f"[Stage 4B] Synthesizing & Whisper-aligning {len(tts_items)} cloned voiceover segments on RTX Pro 6000...")
            jid_tts = client.exec_async(
                f"python3 -u rtx6000_heavy_worker.py tts-batch --manifest ../jobs/{run_dir.name}/tts_manifest.json --profile ../voices/profile/voice_profile.json --out-dir ../jobs/{run_dir.name}/tts",
                cwd="repo",
                env=remote_hf_env,
                job_id=f"tts_{run_dir.name}",
            )
            client.wait_job(jid_tts, poll_sec=4.0, max_wait_sec=2400.0)
            client.get_tar_dir(f"jobs/{run_dir.name}/tts", tts_local_dir)
            print(f"  -> Downloaded all {len(tts_items)} broadcast WAVs + Whisper caption alignments to {tts_local_dir}!")
        else:
            # Try Tailscale SSH `b850` RTX 5070 Ti CUDA in isolated `/dev/shm/heydigital_g4c` first (if >= 3500 MiB free), else local MPS!
            import subprocess
            import sys
            ssh_cfg = REPO_ROOT / "config" / "ssh_config"
            b850_ok = False
            try:
                chk = subprocess.check_output(
                    ["ssh", "-F", str(ssh_cfg), "-o", "ConnectTimeout=4", "b850",
                     "/dev/shm/heydigital_g4c/.venv/bin/python -c 'import torch; print(torch.cuda.mem_get_info(0)[0]//1048576)' 2>/dev/null"],
                    text=True, timeout=8,
                ).strip()
                if chk.isdigit() and int(chk) >= 3400:
                    b850_ok = True
            except Exception:
                b850_ok = False

            if b850_ok:
                print(f"[Stage 4B] Synthesizing & Whisper-aligning {len(tts_items)} segments on Tailscale `b850` RTX 5070 Ti CUDA (/dev/shm RAM-disk)...")
                r_job = f"/dev/shm/heydigital_g4c/jobs/{run_dir.name}"
                subprocess.run(["ssh", "-F", str(ssh_cfg), "b850", f"mkdir -p {r_job}/tts /dev/shm/heydigital_g4c/voices"], check=True)
                vp_local = json.loads((local_vp_dir / "voice_profile.json").read_text(encoding="utf-8"))
                ref_wav_local = REPO_ROOT / vp_local["ref_wav"] if not Path(vp_local["ref_wav"]).is_absolute() else Path(vp_local["ref_wav"])
                if not ref_wav_local.exists():
                    ref_wav_local = local_vp_dir / "alan_v6_long_m5_24k.wav"
                vp_remote = dict(vp_local)
                vp_remote["ref_wav"] = f"/dev/shm/heydigital_g4c/voices/{ref_wav_local.name}"
                subprocess.run([
                    "scp", "-q", "-F", str(ssh_cfg),
                    str(ref_wav_local),
                    f"b850:/dev/shm/heydigital_g4c/voices/{ref_wav_local.name}",
                ], check=True)
                if vp_local.get("short_ref_wav"):
                    s_wav_local = REPO_ROOT / vp_local["short_ref_wav"] if not Path(vp_local["short_ref_wav"]).is_absolute() else Path(vp_local["short_ref_wav"])
                    if s_wav_local.exists():
                        vp_remote["short_ref_wav"] = f"/dev/shm/heydigital_g4c/voices/{s_wav_local.name}"
                        subprocess.run([
                            "scp", "-q", "-F", str(ssh_cfg),
                            str(s_wav_local),
                            f"b850:/dev/shm/heydigital_g4c/voices/{s_wav_local.name}",
                        ], check=True)
                tmp_vp = run_dir / "_b850_vp.json"
                tmp_vp.write_text(json.dumps(vp_remote, ensure_ascii=False, indent=2), encoding="utf-8")
                subprocess.run([
                    "scp", "-q", "-F", str(ssh_cfg),
                    str(REPO_ROOT / "worker" / "rtx6000_heavy_worker.py"),
                    "b850:/dev/shm/heydigital_g4c/rtx6000_heavy_worker.py",
                ], check=True)
                subprocess.run([
                    "scp", "-q", "-F", str(ssh_cfg),
                    str(tmp_vp),
                    "b850:/dev/shm/heydigital_g4c/voices/voice_profile.json",
                ], check=True)
                subprocess.run([
                    "scp", "-q", "-F", str(ssh_cfg),
                    str(run_dir / "tts_manifest.json"),
                    f"b850:{r_job}/tts_manifest.json",
                ], check=True)
                tmp_vp.unlink(missing_ok=True)
                subprocess.run([
                    "ssh", "-F", str(ssh_cfg), "b850",
                    f"HF_HOME=/tmp/heydigital_hf TMPDIR=/tmp /dev/shm/heydigital_g4c/.venv/bin/python -u /dev/shm/heydigital_g4c/rtx6000_heavy_worker.py tts-batch "
                    f"--manifest {r_job}/tts_manifest.json --profile /dev/shm/heydigital_g4c/voices/voice_profile.json --out-dir {r_job}/tts",
                ], check=True)
                tts_local_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    "bash", "-c",
                    f"ssh -F '{ssh_cfg}' b850 'tar -czf - -C {r_job}/tts .' | tar -xzf - -C '{tts_local_dir}' && ssh -F '{ssh_cfg}' b850 'rm -rf {r_job}'",
                ], check=True)
                print(f"  -> Pulled all {len(tts_items)} studio WAVs + Whisper alignments from `b850` to local {tts_local_dir}!")
            else:
                print(f"[Stage 4B] Synthesizing {len(tts_items)} segments locally via 2-way parallel Apple Metal MPS workers...")
                tts_local_dir.mkdir(parents=True, exist_ok=True)
                vp_local_abs = json.loads((local_vp_dir / "voice_profile.json").read_text(encoding="utf-8"))
                vp_local_abs["ref_wav"] = str((REPO_ROOT / vp_local_abs["ref_wav"]).resolve())
                tmp_vp = run_dir / "_local_vp.json"
                tmp_vp.write_text(json.dumps(vp_local_abs, ensure_ascii=False, indent=2), encoding="utf-8")

                n_workers = 2
                procs = []
                shard_dirs = []
                for wi in range(n_workers):
                    s_items = tts_items[wi::n_workers]
                    s_mf = run_dir / f"_tts_mf_s{wi}.json"
                    s_out = run_dir / f"_tts_out_s{wi}"
                    s_out.mkdir(parents=True, exist_ok=True)
                    s_mf.write_text(json.dumps({"speed": 1.01, "short_speed": 1.06, "items": s_items}, ensure_ascii=False), encoding="utf-8")
                    shard_dirs.append((s_mf, s_out))
                    procs.append(subprocess.Popen([
                        sys.executable, "-u", str(REPO_ROOT / "worker" / "rtx6000_heavy_worker.py"),
                        "tts-batch", "--manifest", str(s_mf), "--profile", str(tmp_vp), "--out-dir", str(s_out),
                    ]))
                for p_w in procs:
                    if p_w.wait() != 0:
                        raise RuntimeError("Local MPS TTS worker shard failed")

                import shutil
                merged_map = {}
                for s_mf, s_out in shard_dirs:
                    s_res = json.loads((s_out / "tts_results.json").read_text(encoding="utf-8"))
                    for it in s_res["items"]:
                        merged_map[it["id"]] = it
                        shutil.move(str(s_out / it["wav"]), str(tts_local_dir / it["wav"]))
                    s_mf.unlink(missing_ok=True)
                    shutil.rmtree(s_out, ignore_errors=True)
                ordered_items = [merged_map[it["id"]] for it in tts_items]
                (tts_local_dir / "tts_results.json").write_text(
                    json.dumps({"ok": True, "gpu": "Apple Silicon Metal MPS x2", "count": len(ordered_items), "items": ordered_items}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp_vp.unlink(missing_ok=True)

    # Step 4C: Spotlight Thumbnail Hero (`RealVisXL_V4.0`) + Chapter Card Silhouettes (all B-roll uses real SigLIP-matched footage!)
    img_local_dir = run_dir / "images"
    if not (img_local_dir / "image_results.json").exists():
        img_items = []
        th_prompt = (plan.get("thumbnail") or {}).get("image_prompt") or "Minimalist dark studio spotlight product shot of sleek silver and space black aluminum laptops side by side, rim lighting, matte charcoal background, no text, 8k"
        img_items.append({"id": "THUMB_HERO", "prompt": th_prompt, "width": 1344, "height": 768, "seed": 8800, "animate_i2v": False})

        for idx, b in enumerate(plan.get("long_beats", [])):
            if b.get("visual_type") == "chapter_dark_card":
                img_items.append({
                    "id": b["id"],
                    "prompt": th_prompt,
                    "width": 1280,
                    "height": 720,
                    "seed": 5100 + idx,
                    "animate_i2v": False,
                })

        img_manifest = {"enable_i2v": False, "items": img_items}
        (run_dir / "img_manifest.json").write_text(json.dumps(img_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        client.put_file(run_dir / "img_manifest.json", f"jobs/{run_dir.name}/img_manifest.json")

        print(f"[Stage 4C] Generating {len(img_items)} studio stills + LTX-Video I2V motion clips on RTX Pro 6000...")
        jid_img = client.exec_async(
            f"python3 -u rtx6000_heavy_worker.py img-batch --manifest ../jobs/{run_dir.name}/img_manifest.json --out-dir ../jobs/{run_dir.name}/images",
            cwd="repo",
            env=remote_hf_env,
            job_id=f"img_{run_dir.name}",
        )
        client.wait_job(jid_img, poll_sec=4.0, max_wait_sec=2400.0)
        client.get_tar_dir(f"jobs/{run_dir.name}/images", img_local_dir)
        print(f"  -> Downloaded all generated stills & I2V clips to local {img_local_dir}!")

    # Step 5: Local Metal WebGL2 Three.js + Interactive Short + Rhodes Sidechain Master Rendering
    deliverables_dir = run_dir / "deliverables"
    deliverables_dir.mkdir(parents=True, exist_ok=True)

    hero_png = img_local_dir / "THUMB_HERO.png"
    hero_im = Image.open(hero_png).convert("RGB") if hero_png.exists() else None
    thumb_out = deliverables_dir / "thumbnail_1080p.png"
    render_thumbnail(plan, hero_im, thumb_out)
    print(f"[Stage 5A] Thumbnail rendered: {thumb_out}")

    long_out = deliverables_dir / "hey_digital_long_1080p.mp4"
    long_info = assemble_long_video(
        plan=plan,
        tts_dir=tts_local_dir,
        img_dir=img_local_dir,
        out_mp4=long_out,
        work_dir=run_dir / "work_long",
        broll_dir=broll_dir,
        repo_root=REPO_ROOT,
        run_dir=run_dir,
    )
    (run_dir / "long_render_info.json").write_text(json.dumps(long_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Stage 5B] V2 Long-Form Video rendered: {long_out} ({long_info['duration']}s)")

    short_out = deliverables_dir / "hey_digital_short_funnel_1080x1920.mp4"
    short_info = assemble_short_video(
        plan=plan,
        tts_dir=tts_local_dir,
        img_dir=img_local_dir,
        thumb_path=thumb_out,
        out_mp4=short_out,
        work_dir=run_dir / "work_short",
        broll_dir=broll_dir,
        repo_root=REPO_ROOT,
        run_dir=run_dir,
    )
    print(f"[Stage 5C] V2 Interactive Traffic-Funnel Short rendered: {short_out} ({short_info['duration']}s)")

    desc_long, desc_short = format_youtube_descriptions(plan, long_info["chapters"])
    (deliverables_dir / "youtube_long_metadata.txt").write_text(
        f"TITLE:\n{plan.get('title_long', '')}\n\nDESCRIPTION:\n{desc_long}\n", encoding="utf-8"
    )
    (deliverables_dir / "youtube_short_metadata.txt").write_text(
        f"TITLE:\n{plan.get('title_short', '')}\n\nRELATED LONG VIDEO:\n{plan.get('title_long', '')}\n\nDESCRIPTION:\n{desc_short}\n", encoding="utf-8"
    )
    summary = {
        "pipeline_version": "6.0-qwen3-tts-bilingual-precision",
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
    if getattr(args, "send_tg", False):
        from hey_digital_pipeline.tg_delivery import send_run_to_telegram
        tg_res = send_run_to_telegram(deliverables_dir)
        print(f"[Stage 6] Delivered Thumbnail + Long + Short to Telegram: {tg_res}")
    print(f"=== V2 PIPELINE COMPLETE! Deliverables saved locally to: {deliverables_dir} ===")
    return deliverables_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Hey Digital Gemini 4 Carbon V2 — Autonomous Video & Shorts Pipeline")
    ap.add_argument("--topic", default="", help="Optional manual topic override; leave empty for 100%% unattended topic discovery")
    ap.add_argument("--queries", default="", help="Comma-separated search queries when --topic is provided")
    ap.add_argument("--video-type", default="versus_guide", choices=["versus_guide", "ai_benchmark", "deep_review", "hidden_tips", "news_analysis"])
    ap.add_argument("--long-beats", type=int, default=28, help="Target number of narrative/visual beats for Long-form video")
    ap.add_argument("--short-steps", type=int, default=5, help="Target number of structured steps in Traffic-Funnel Short")
    ap.add_argument("--colab-url", default="", help="HTTPS trycloudflare URL for RTX Pro 6000 bridge")
    ap.add_argument("--colab-token", default="", help="Bearer token for RTX Pro 6000 bridge")
    ap.add_argument("--run-id", default="", help="Custom run folder name under runs/")
    ap.add_argument("--skip-i2v", action="store_true", help="Skip LTX-Video I2V stage if quick turnaround is desired")
    ap.add_argument("--send-tg", action="store_true", help="Automatically push finished Thumbnail, Long MP4, and Short MP4 to Telegram")
    ap.add_argument("--force-regen-script", action="store_true")
    return ap


if __name__ == "__main__":
    run_pipeline(build_arg_parser().parse_args())
