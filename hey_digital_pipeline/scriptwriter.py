#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon — Crawled-Fact AI Script & Storyboard Generator (Local Master)
==========================================================================================
Modular Chapter-by-Chapter + Funnel-Short Storyboard Architecture:
- Generates Outline & Thumbnail first, then writes each chapter (4 rich beats/chapter)
  independently with full context of crawled public sources (`[SRC-01]..[SRC-12]`),
  preventing LLM output truncation/timeouts while producing deeper, broadcast-grade
  Hey Digital scripts!

LLM Auto-Cascading Backends:
1. Google Gemini API (`GEMINI_API_KEY` / `GOOGLE_API_KEY`)
2. Cluster OpenAI-compatible endpoint (`HEYDIGITAL_CLUSTER_LLM_URL`)
3. OpenAI / DeepSeek API (`OPENAI_API_KEY` / `DEEPSEEK_API_KEY` — skips automatically on 402)
4. Local Apple Silicon `Ollama` (`http://127.0.0.1:11434/api/chat`, `qwen3:8b`, `think: False`)
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request


VIDEO_TYPE_PROFILES = {
    "versus_guide": "雙機或多陣營深度選購對決（拆解帳面參數迷思、真實使用場景摩擦、隱藏成本經濟帳、明確購買分流建議）",
    "ai_benchmark": "本機 AI 部署與硬體算力實測（拆解記憶體容量 vs 頻寬天花板、Prefill 與 Decode 差異、模型梯隊實測與選購建議）",
    "deep_review": "旗艦科技產品深度評測（工業設計與面板光學進化、晶片刀法與散熱續航取捨、生態定位與目標受眾）",
    "hidden_tips": "必做關鍵設定與效率實戰指南（打破預設耗電或畫質陷阱、逐步實操路徑、前後體感差異對比）",
    "news_analysis": "最新科技硬體與趨勢深度拆解（從公開財報/跑分/供應鏈規格看懂廠商定價邏輯與刀法真相）",
}

_SKIP_PAID_LLM = False


def _extract_json_block(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    else:
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and j > i:
            text = text[i : j + 1]
    return json.loads(text)


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.38, max_tokens: int = 2200) -> dict:
    global _SKIP_PAID_LLM
    ctx = ssl.create_default_context()
    errors = []

    # 1. Gemini API
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        try:
            model = os.environ.get("HEYDIGITAL_GEMINI_MODEL", "gemini-2.5-pro")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
            }
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _extract_json_block(data["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

    # 2. RTX Pro 6000 Heavy Worker LLM (Qwen2.5-14B-Instruct CUDA on 127.0.0.1:19465 via bridge)
    bridge_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "rtx6000_bridge.json")
    if os.path.exists(bridge_file) and os.environ.get("HEYDIGITAL_DISABLE_RTX_LLM") != "1":
        try:
            from pathlib import Path
            from hey_digital_pipeline.remote_worker import ColabBridgeConfig, ColabRTX6000Client
            bcfg = json.loads(Path(bridge_file).read_text(encoding="utf-8"))
            client = ColabRTX6000Client(ColabBridgeConfig(url=bcfg["url"], token=bcfg["token"]))
            req_obj = {
                "messages": [
                    {"role": "system", "content": system_prompt + "\n嚴格只輸出純 JSON 物件，不要加任何解釋。"},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            client.put_bytes(json.dumps(req_obj, ensure_ascii=False).encode("utf-8"), "jobs/llm_req.json")
            jid = client.exec_async(
                "curl -fsS -X POST http://127.0.0.1:19465/v1/chat/completions -H 'Content-Type: application/json' --data-binary @../jobs/llm_req.json -o ../jobs/llm_resp.json",
                cwd="repo",
            )
            client.wait_job(jid, poll_sec=1.5, max_wait_sec=120.0)
            tmp_resp = Path(bridge_file).parent.parent / "runs" / ".tmp_llm_resp.json"
            client.get_file("jobs/llm_resp.json", tmp_resp)
            rdata = json.loads(tmp_resp.read_text(encoding="utf-8"))
            return _extract_json_block(rdata["choices"][0]["message"]["content"])
        except Exception as exc:
            errors.append(f"RTX6000-LLM: {exc}")

    # 2B. Cluster OpenAI-compatible endpoint
    cluster_llm = os.environ.get("HEYDIGITAL_CLUSTER_LLM_URL")
    if cluster_llm:
        try:
            model = os.environ.get("HEYDIGITAL_CLUSTER_LLM_MODEL", "default")
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            req = urllib.request.Request(
                f"{cluster_llm.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('HEYDIGITAL_CLUSTER_LLM_KEY', 'EMPTY')}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _extract_json_block(data["choices"][0]["message"]["content"])
        except Exception as exc:
            errors.append(f"ClusterLLM: {exc}")

    # 3. OpenAI / DeepSeek API (memorize 402/401 so subsequent calls don't waste round-trips)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if api_key and not _SKIP_PAID_LLM:
        try:
            is_ds = bool(os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"))
            base = os.environ.get("HEYDIGITAL_LLM_BASE_URL") or ("https://api.deepseek.com/v1" if is_ds else "https://api.openai.com/v1")
            model = os.environ.get("HEYDIGITAL_LLM_MODEL") or ("deepseek-chat" if is_ds else "gpt-4.1")
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            req = urllib.request.Request(
                f"{base.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _extract_json_block(data["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as hexc:
            if hexc.code in (401, 402, 403, 429):
                _SKIP_PAID_LLM = True
            errors.append(f"OpenAI/DeepSeek: {hexc}")
        except Exception as exc:
            errors.append(f"OpenAI/DeepSeek: {exc}")

    # 4. Local Apple Silicon Ollama (`qwen3:8b`, fast non-thinking mode without slow GBNF grammar lock)
    ollama_url = os.environ.get("HEYDIGITAL_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    ollama_model = os.environ.get("HEYDIGITAL_OLLAMA_MODEL", "qwen3:8b")
    for attempt in range(2):
        try:
            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt + "\n請只輸出純 JSON 物件，不要加任何額外說明文字。 /no_think"},
                    {"role": "user", "content": user_prompt + "\n/no_think"},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": max_tokens},
            }
            if attempt == 1:
                payload["format"] = "json"
            req = urllib.request.Request(
                ollama_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=150) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _extract_json_block(data["message"]["content"])
        except Exception as exc:
            errors.append(f"Ollama-try{attempt+1}: {exc}")

    raise RuntimeError("All LLM backends failed: " + " | ".join(errors))


def formulate_topic_from_candidates(candidates_report: dict, existing_titles: list[str], video_type: str = "versus_guide") -> dict:
    sys_p = (
        "你是 YouTube 科技深度頻道『Hey Digital (@hey_digitall)』的總編導 Agent。"
        "頻道風格：台灣繁體中文（記憶體、頻寬、晶片、螢幕、筆電、硬碟、演算法），理性犀利拆解 Apple / AI PC / GPU / 行動硬體的規格刀法與選購痛點。"
        "請根據最新爬蟲候選新聞，自動選定一個最適合本期更新、具高度觀看吸引力的主題，並給出精準搜尋關鍵字。只回傳 JSON。"
    )
    usr_p = json.dumps({
        "target_video_type": video_type,
        "already_published_titles": existing_titles[-8:],
        "live_crawled_candidates": candidates_report.get("top_candidates", [])[:6],
        "required_json": {
            "selected_topic": "繁體中文核心選題標題（帶疑問或反差痛點）",
            "video_type": video_type,
            "angle_rationale": "選題理由",
            "search_queries": ["英文檢索詞1", "英文檢索詞2", "繁中檢索詞3"],
            "wiki_terms": ["MacBook Air", "MacBook Pro"],
        },
    }, ensure_ascii=False)
    res = call_llm_json(sys_p, usr_p, temperature=0.35, max_tokens=900)
    if not res.get("selected_topic"):
        res["selected_topic"] = "MacBook Air M5 對決 MacBook Pro M6：價差與效能刀法背後，買哪台才不後悔？"
    return res


def split_into_captions(narration: str, max_len: int = 19) -> list[str]:
    parts = re.split(r"[，。！？；：、\n]+", narration.strip())
    out = []
    for p in parts:
        p = p.strip(" 「」『』\"'，。！？；：")
        if not p:
            continue
        p = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", p)
        p = re.sub(r"([A-Za-z0-9%])([\u4e00-\u9fff])", r"\1 \2", p)
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) <= max_len:
            out.append(p)
        else:
            sp = p.rfind(" ", 4, min(len(p) - 3, max_len))
            if sp != -1:
                out.append(p[:sp].strip())
                out.append(p[sp + 1 :].strip())
            else:
                out.append(p[:max_len].strip())
                if p[max_len:].strip():
                    out.append(p[max_len:].strip())
    return [x for x in out if x]


def generate_production_plan(
    topic: str,
    video_type: str,
    dossier: dict,
    target_long_beats: int = 16,
    target_short_steps: int = 5,
) -> dict:
    """Modular chapter-by-chapter + funnel-short storyboard generation."""
    sources_compact = [
        {
            "source_id": s["source_id"],
            "publisher": s["publisher"],
            "domain": s["domain"],
            "title": s["title"],
            "published": s["published"],
            "summary": s["summary"][:240],
            "metrics": s["extracted_metrics"][:6],
        }
        for s in dossier.get("sources", [])[:8]
    ]
    valid_sids = [s["source_id"] for s in sources_compact] or ["SRC-01"]

    # Step 1: Outline (Title + Summary + Thumbnail + 4 Chapters)
    print("  [Scriptwriter] Step 1/3: Generating outline & thumbnail...")
    outline = call_llm_json(
        "你是『Hey Digital (@hey_digitall)』科技頻道總編導。使用道地台灣繁體中文，設計長片標題、縮圖雙行金句與 4 個深度拆解章節。只輸出 JSON。",
        json.dumps({
            "topic": topic,
            "video_type": video_type,
            "sources_overview": [{"id": s["source_id"], "title": s["title"], "metrics": s["metrics"]} for s in sources_compact],
            "schema": {
                "title_long": "具反差張力的繁體中文長片標題",
                "summary_paragraph": "100字內精華摘要",
                "thumbnail": {
                    "line1": "縮圖上行大字(6~9字)",
                    "line2": "縮圖下行大字(6~9字)",
                    "gold_phrase": "下行或上行中要標為香檳金的關鍵詞(3~5字)",
                    "image_prompt": "Minimalist dark studio spotlight product shot of sleek aluminum MacBook Air and MacBook Pro laptops with glowing Apple silicon chip, dramatic rim light, matte black background, 8k, no text",
                },
                "chapters": [
                    {"num": "01", "title": "價格與定位陷阱", "en": "PRICING PARADOX", "focus": "拆解最新降價後兩台機器的真實入手門檻差距"},
                    {"num": "02", "title": "晶片與散熱刀法", "en": "THERMAL & SILICON", "focus": "無風扇被動散熱與主動散熱在高負載下的頻寬與持續效能差異"},
                    {"num": "03", "title": "螢幕與外接生態", "en": "DISPLAY & PORTS", "focus": "高刷面板、峰值亮度與多螢幕外接連接埠的隱形體驗落差"},
                    {"num": "04", "title": "終極購買建議", "en": "BUYING VERDICT", "focus": "按真實工作流給出明確不踩雷的配置與容量選購結論"},
                ],
            },
        }, ensure_ascii=False),
        temperature=0.38,
        max_tokens=1200,
    )

    chapters = (outline.get("chapters") or [])[:4]
    if len(chapters) < 4:
        chapters = [
            {"num": "01", "title": "價格與定位反差", "en": "PRICING PARADOX", "focus": "拆解最新優惠價差與容量升級陷阱"},
            {"num": "02", "title": "晶片頻寬與散熱", "en": "SILICON & THERMALS", "focus": "對比持續高負載與 AI 推論時的真實效能"},
            {"num": "03", "title": "螢幕亮度與周邊", "en": "DISPLAY & IO", "focus": "面板刷新率、亮度尼特數與連接埠差異"},
            {"num": "04", "title": "誰該買哪一台", "en": "BUYING GUIDE", "focus": "針對不同預算與使用族群給出精準推薦"},
        ]

    # Step 2: Generate 4 rich beats per chapter (16 beats total)
    all_long_beats = []
    beats_per_ch = max(3, target_long_beats // len(chapters))
    for ch_idx, ch in enumerate(chapters):
        ch_num = str(ch.get("num") or f"{ch_idx+1:02d}").zfill(2)
        print(f"  [Scriptwriter] Step 2/3: Writing Chapter {ch_num} ({ch.get('title')})...")
        ch_res = call_llm_json(
            "你是『Hey Digital』科技主筆。使用台灣繁體中文（記憶體、頻寬、晶片、螢幕、筆電、硬碟）。"
            f"請為第 {ch_num} 章撰寫恰好 {beats_per_ch} 個鏡次 (`beats`)：\n"
            "- 第 1 鏡固定 `visual_type: \"chapter_dark_card\"`（破題口播 22~35 字）。\n"
            "- 第 2 鏡固定 `visual_type: \"studio_isometric_infographic\"`（暖白 3D 數據圖卡，給出具體數字 `hero_metric`、`hero_unit`、3 組 `comparison_bars`，口播 48~72 字引用公開數據）。\n"
            "- 第 3~4 鏡使用 `\"cinema_tech_broll\"` 或 `\"studio_isometric_infographic\"`（口播 48~72 字，附英文商業硬體攝影 `image_prompt` 且無文字浮水印，引用 `source_id`）。\n"
            "只輸出 JSON。",
            json.dumps({
                "video_title": outline.get("title_long", topic),
                "chapter": ch,
                "valid_source_ids": valid_sids,
                "crawled_sources": sources_compact,
                "schema": {
                    "beats": [
                        {
                            "visual_type": "chapter_dark_card",
                            "narration": "...",
                            "source_id": "SRC-01",
                            "hero_metric": "300",
                            "hero_unit": "USD",
                            "accent_hex": "#B84A14",
                            "comparison_bars": [
                                {"label": "高配機型", "value": 92, "display": "120Hz ProMotion"},
                                {"label": "輕薄機型", "value": 58, "display": "60Hz Liquid Retina"},
                            ],
                            "image_prompt": "Photorealistic walnut wood slat desk with silver MacBook Air and Space Black MacBook Pro side by side, warm studio rim lighting, 35mm cinema photo, no text",
                            "overlay_news_card": True,
                        }
                    ]
                },
            }, ensure_ascii=False),
            temperature=0.4,
            max_tokens=1800,
        )
        for b in (ch_res.get("beats") or [])[:beats_per_ch]:
            b["chapter_num"] = ch_num
            b["chapter_title"] = ch.get("title", "核心拆解")
            b["chapter_en"] = ch.get("en", "DEEP DIVE")
            all_long_beats.append(b)

    # Step 3: Generate Corresponding Traffic-Funnel Short (7 beats: Hook + 5 Steps + Funnel CTA)
    print("  [Scriptwriter] Step 3/3: Writing corresponding Traffic-Funnel Short...")
    short_plan = call_llm_json(
        "你是『Hey Digital』YouTube Shorts 導流短片導演。使用台灣繁體中文，為這支長片製作恰好 7 鏡的精準推流短片 (`short_beats`)：\n"
        "- 第 1 鏡：`visual_type: \"hook_sticker_broll\"`，含 `sticker_top`(5~7字)、`sticker_bottom`(4~6字)、`narration`(24~35字)、英文直式攝影 `image_prompt`。\n"
        "- 第 2~6 鏡（5 個精華步驟）：`visual_type: \"step_phone_mockup\"`，含 `topic_pill`、`step_title`(4~6字)、`flow_steps`(3個短詞陣列)、`callout_tag`(4字)、`card_metric`、`card_points`(2條重點)、`source_id`、`narration`(28~42字)、英文 `image_prompt`。\n"
        "- 第 7 鏡：`visual_type: \"cta_funnel_endcard\"`，口播明確引導觀眾點擊下方關聯長片收看完整實測與全部公開來源。\n"
        "只輸出 JSON。",
        json.dumps({
            "parent_long_title": outline.get("title_long", topic),
            "summary": outline.get("summary_paragraph", ""),
            "valid_source_ids": valid_sids,
            "sources": sources_compact[:5],
            "schema": {
                "title_short": "高點擊繁中導流標題 #Shorts",
                "short_beats": [
                    {
                        "visual_type": "hook_sticker_broll",
                        "topic_pill": "筆電選購避坑",
                        "sticker_top": "降價三百美元",
                        "sticker_bottom": "該直上 Pro 嗎？",
                        "step_title": "價差陷阱",
                        "flow_steps": ["看基準容量", "算升級差價", "選最佳甜蜜點"],
                        "callout_tag": "必看刀法",
                        "card_metric": "差價不到兩百刀",
                        "card_points": ["入門記憶體容量關鍵差異", "外接螢幕與高刷體感落差"],
                        "source_id": "SRC-01",
                        "narration": "...",
                        "image_prompt": "Vertical close up studio shot of sleek aluminum MacBook laptop on dark desk, warm rim lighting, no text",
                    }
                ],
            },
        }, ensure_ascii=False),
        temperature=0.38,
        max_tokens=2000,
    )

    # Normalize & link verifiable crawled sources
    src_map = {s["source_id"]: s for s in dossier.get("sources", [])}
    fallback_src = dossier["sources"][0] if dossier.get("sources") else {
        "source_id": "SRC-01",
        "publisher": "Public Tech Spec",
        "domain": "9to5mac.com",
        "title": topic,
        "url": "https://9to5mac.com/",
        "published": "2026",
        "summary": topic,
        "extracted_metrics": [],
    }
    used_sources = {}
    accents = ["#B84A14", "#0E7C47", "#0E6EB8", "#5B50D6"]

    for idx, b in enumerate(all_long_beats, start=1):
        b["id"] = f"L{idx:02d}"
        if b.get("visual_type") not in ("chapter_dark_card", "studio_isometric_infographic", "cinema_tech_broll"):
            b["visual_type"] = "studio_isometric_infographic" if idx % 2 == 0 else "cinema_tech_broll"
        b["accent_hex"] = b.get("accent_hex") or accents[idx % len(accents)]
        if not b.get("image_prompt"):
            b["image_prompt"] = "Photorealistic studio desk setup with aluminum MacBook Air and MacBook Pro, walnut wood slat wall, warm cinematic lighting, 8k, no text"
        sid = b.get("source_id") if b.get("source_id") in src_map else valid_sids[(idx - 1) % len(valid_sids)]
        b["source_id"] = sid
        s_obj = src_map.get(sid, fallback_src)
        used_sources[sid] = s_obj
        b["source_detail"] = s_obj
        dom = s_obj.get("domain", "")
        tag_sub = s_obj.get("published", "")[:10] if "news.google" in dom else dom
        b["source_badge"] = f"資料來源：{s_obj['publisher']} ({tag_sub})"
        b["captions"] = split_into_captions(b.get("narration", ""), max_len=19)

    short_beats = short_plan.get("short_beats") or []
    if short_beats:
        short_beats[0]["visual_type"] = "hook_sticker_broll"
        if short_beats[-1].get("visual_type") != "cta_funnel_endcard":
            short_beats.append({
                "visual_type": "cta_funnel_endcard",
                "narration": "想看完整四個維度的深度實測對比與全部公開數據來源，現在就點擊下方關聯影片，收看完整長片解析！",
                "source_id": fallback_src["source_id"],
            })
    for idx, sb in enumerate(short_beats, start=1):
        sb["id"] = f"S{idx:02d}"
        if 1 < idx < len(short_beats):
            sb["visual_type"] = "step_phone_mockup"
        sid = sb.get("source_id") if sb.get("source_id") in src_map else valid_sids[(idx - 1) % len(valid_sids)]
        sb["source_id"] = sid
        s_obj = src_map.get(sid, fallback_src)
        used_sources[sid] = s_obj
        sb["source_detail"] = s_obj
        if not sb.get("image_prompt"):
            sb["image_prompt"] = "Vertical studio close-up of sleek modern aluminum MacBook laptop and glowing Apple silicon chip on walnut desk, rim lighting, no text"
        sb["captions"] = split_into_captions(sb.get("narration", ""), max_len=15)

    for s in dossier.get("sources", [])[:6]:
        used_sources.setdefault(s["source_id"], s)

    return {
        "topic": topic,
        "video_type": video_type,
        "title_long": outline.get("title_long") or topic,
        "title_short": short_plan.get("title_short") or f"{topic[:18]} #Shorts",
        "summary_paragraph": outline.get("summary_paragraph") or topic,
        "thumbnail": outline.get("thumbnail") or {
            "line1": "Air M5 對決 Pro M6",
            "line2": "降價後買誰才不虧？",
            "gold_phrase": "買誰才不虧",
            "image_prompt": "Minimalist dark studio spotlight product shot of sleek aluminum MacBook Air and MacBook Pro laptops, matte black background, 8k, no text",
        },
        "long_beats": all_long_beats,
        "short_beats": short_beats,
        "cited_sources": list(used_sources.values()),
    }
