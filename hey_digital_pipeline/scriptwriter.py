#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon V2 — Flagship 32B/Gemini AI Script & High-Density Storyboard Director
==================================================================================================
V2 Upgrades matched 1:1 against `@hey_digitall` verbatim transcripts (`D9IiQF_QIYg`, `1TO7dzrTuQg`):
1. Authentic `@hey_digitall` Few-Shot Narrative Style:
   - Conversational, razor-sharp Taiwanese hardware storytelling ("如果你這幾年一直想換…大概都被蘋果那個經典的容量升級策略折磨過…攤開底層硬體規格來看…").
   - Strictly forbids robotic encyclopedia filler ("根據最新數據，X 配備了 X 處理器，提供卓越效能").
2. High-Density Visual Choreography (28–36 Long Beats + 9–10 Interactive Short Beats):
   - Every 5–8 seconds switches between **8 authentic Hey Digital visual templates**:
     * `chapter_dark_card` (1.8s crisp silent/musical chapter transition card — just like `1TO7dzrTuQg`!)
     * `webgl_memory_lanes` (Three.js 3D isometric dual/triple bandwidth rails with morphing metric `170 -> 153 -> -10%`)
     * `webgl_ssd_particles` (Three.js 3D NAND grid with flying parabolic data cubes into 3D laptop)
     * `webgl_stacked_tco` (Three.js 3D camera pan from laptop to stacked `BASE vs PRO` upgrade price blocks)
     * `webgl_chip_die` (Three.js 3D silicon die package with glowing CPU/GPU/NPU core clusters)
     * `webgl_dual_laptop` (Three.js 3D dual laptop speed race / thermal heat-glow comparison)
     * `ui_configurator_crawl` (Animated Apple-style price configurator cursor click OR live crawled media article spotlight)
     * `cinema_broll_video` (Real harvested 1080p hardware hands-on B-roll video OR RTX Pro 6000 LTX-Video I2V cinema motion)
3. Dual Text Output + Fact-Check Purifier:
   - Produces clean `narration` (for display/captions) AND `tts_text` (100% pure Mandarin spoken form via `cn_verbalizer`, zero Arabic digits).
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from hey_digital_pipeline.cn_verbalizer import purify_tw_tech_text, verbalize_for_mandarin_tts

VIDEO_TYPE_PROFILES = {
    "versus_guide": "雙機或多陣營深度選購對決（拆解容量與記憶體升級刀法、真實高負載摩擦、外接與螢幕妥協、TCO 經濟帳與明確分流建議）",
    "ai_benchmark": "本機 AI 部署與硬體算力實測（拆解統一記憶體容量 vs 頻寬天花板、Prefill 與 Decode 本質差異、模型梯隊實戰與選購指南）",
    "deep_review": "旗艦科技產品深度評測（工業設計與面板光學進化、晶片分檔刀法與散熱續航取捨、生態定位與目標受眾）",
    "hidden_tips": "必做關鍵設定與效率實戰指南（打破預設耗電或畫質陷阱、逐步實操路徑、前後體感差異對比）",
    "news_analysis": "最新科技硬體與趨勢深度拆解（從公開規格、跑分與定價階梯看懂廠商刀法真相）",
}

HEY_DIGITAL_FEWSHOT_ANCHOR = """
【Hey Digital 原頻道真實口播範例（請嚴格模仿此語氣、節奏、用詞與敘事張力，嚴禁寫成生硬說明書）】：
範例 1（開場痛點共鳴）：
「如果你這幾年一直想換一台輕薄筆電，大概都被蘋果那個經典的容量升級策略折磨過：買最便宜的入門版，硬碟空間捉襟見肘，用不到兩年就天天跳出空間不足的警告；可是一旦想加錢升級記憶體和硬碟，價格瞬間就逼近高階款，讓人陷入無止境的糾結。不過當你真正把它當成主力機高強度使用幾個禮拜之後，心裡反而會冒出一種非常微妙的反差感。」
範例 2（底層規格刀法拆解）：
「攤開底層硬體規格來看，記憶體頻寬竟然會隨著容量不同而分級：如果你買的是 24GB 或 32GB 版本，記憶體頻寬可以跑滿 170 GB/s；但如果你買的是入門的 16GB 版本，頻寬就被砍到了 153 GB/s，硬生生少了大約 10%。再加上 macOS 系統本身開機就要佔用掉大約 4GB 記憶體，扣掉日常開著瀏覽器或通訊軟體的消耗，真正能留給重度任務的空間其實非常吃緊。」
"""

LONG_SCENE_TYPES = [
    "webgl_memory_lanes",
    "webgl_ssd_particles",
    "webgl_stacked_tco",
    "webgl_chip_die",
    "webgl_dual_laptop",
    "ui_configurator_crawl",
    "cinema_broll_video",
]

_SKIP_PAID_LLM = False


def _extract_json_block(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", (text or "").strip(), flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    else:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            text = text[i : j + 1]
    return json.loads(text)


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.42, max_tokens: int = 2800) -> dict:
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
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=150, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _extract_json_block(data["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as exc:
            errors.append(f"Gemini: {exc}")

    # 2. RTX Pro 6000 Flagship 32B/14B CUDA Server (:19465)
    bridge_file = Path(__file__).resolve().parent.parent / "config" / "rtx6000_bridge.json"
    if bridge_file.exists() and os.environ.get("HEYDIGITAL_DISABLE_RTX_LLM") != "1":
        try:
            from hey_digital_pipeline.remote_worker import ColabBridgeConfig, ColabRTX6000Client
            bcfg = json.loads(bridge_file.read_text(encoding="utf-8"))
            client = ColabRTX6000Client(ColabBridgeConfig(url=bcfg["url"], token=bcfg["token"]))
            req_obj = {
                "messages": [
                    {"role": "system", "content": system_prompt + "\n嚴格只輸出合法 JSON 物件，不要任何解釋或 Markdown 註解。"},
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
            client.wait_job(jid, poll_sec=2.0, max_wait_sec=300.0)
            tmp_resp = bridge_file.parent.parent / "runs" / ".tmp_llm_resp.json"
            client.get_file("jobs/llm_resp.json", tmp_resp)
            rdata = json.loads(tmp_resp.read_text(encoding="utf-8"))
            tmp_resp.unlink(missing_ok=True)
            return _extract_json_block(rdata["choices"][0]["message"]["content"])
        except Exception as exc:
            errors.append(f"RTX6000-32B: {exc}")

    # 3. OpenAI / DeepSeek API
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
            with urllib.request.urlopen(req, timeout=90, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return _extract_json_block(data["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as hexc:
            if hexc.code in (401, 402, 403, 429):
                _SKIP_PAID_LLM = True
            errors.append(f"PaidAPI: {hexc}")
        except Exception as exc:
            errors.append(f"PaidAPI: {exc}")

    # 4. Local Ollama fallback
    ollama_url = os.environ.get("HEYDIGITAL_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    ollama_model = os.environ.get("HEYDIGITAL_OLLAMA_MODEL", "qwen3:8b")
    try:
        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt + "\n只輸出 JSON。 /no_think"},
                {"role": "user", "content": user_prompt + "\n/no_think"},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": max_tokens},
        }
        req = urllib.request.Request(ollama_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=360) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return _extract_json_block(data["message"]["content"])
    except Exception as exc:
        errors.append(f"Ollama: {exc}")

    raise RuntimeError("All LLM backends failed: " + " | ".join(errors))


def formulate_topic_from_candidates(candidates_report: dict, existing_titles: list[str], video_type: str = "versus_guide") -> dict:
    sys_p = (
        "你是 YouTube 科技深度頻道『Hey Digital (@hey_digitall)』總編導。\n"
        + HEY_DIGITAL_FEWSHOT_ANCHOR
        + "\n請從最新公開爬蟲候選新聞中，挑出一個最具購買決策張力、真實數據豐富且與頻道舊片不重複的主題，並給出精準英文與繁中檢索關鍵字（同時供網頁爬蟲與實拍 B-roll 搜尋使用）。只回傳 JSON。"
    )
    usr_p = json.dumps({
        "target_video_type": video_type,
        "already_published_titles": existing_titles[-12:],
        "live_crawled_candidates": candidates_report.get("top_candidates", [])[:7],
        "schema": {
            "selected_topic": "繁體中文高張力選題（例如：差價四百美元，買 MacBook Air M4/M5 還是直上 MacBook Pro？實測後發現三個隱藏刀法）",
            "broll_query": "MacBook Air vs MacBook Pro hands on review",
            "search_queries": ["MacBook Air vs MacBook Pro memory bandwidth display comparison", "Apple M4 Pro memory bandwidth SSD speed review", "MacBook Air Pro 選購 頻寬 螢幕 比較"],
            "wiki_terms": ["MacBook Air", "MacBook Pro", "Apple silicon"],
        },
    }, ensure_ascii=False)
    res = call_llm_json(sys_p, usr_p, temperature=0.38, max_tokens=900)
    res["selected_topic"] = purify_tw_tech_text(res.get("selected_topic") or "輕薄 Air 對決全能 Pro：升級記憶體與硬碟後，誰才是真正不踩雷的主力機？")
    if not res.get("broll_query"):
        res["broll_query"] = "MacBook Air vs MacBook Pro review"
    return res


def split_into_captions(narration: str, max_len: int = 18) -> list[str]:
    """
    Split narration into clean, balanced subtitle clauses:
    - Never leaves orphan 1–4 char fragments ("格", "整解析", "GB/s").
    - Never breaks inside ASCII number+unit tokens ("273 GB/s", "256GB SSD", "MacBook Air").
    """
    narration = purify_tw_tech_text(narration)
    parts = re.split(r"[，。！？；：、\n]+", narration)
    out = []
    soft_limit = max_len + 4
    for p in parts:
        p = p.strip(" 「」『』\"'，。！？；：")
        if not p:
            continue
        if len(p) <= soft_limit:
            out.append(p)
            continue
        # Find a safe split point near the middle that is NOT between ASCII/digit characters
        mid = len(p) // 2
        best_idx = -1
        for delta in range(0, max(5, mid - 4)):
            for cand in (mid + delta, mid - delta):
                if 5 <= cand <= len(p) - 5:
                    left_ch, right_ch = p[cand - 1], p[cand]
                    # Only split where at least one side is CJK (never split "273 GB/s" or "Pro Motion")
                    if ("\u4e00" <= left_ch <= "\u9fff") and ("\u4e00" <= right_ch <= "\u9fff"):
                        best_idx = cand
                        break
            if best_idx != -1:
                break
        if best_idx != -1:
            out.append(p[:best_idx].strip())
            out.append(p[best_idx:].strip())
        else:
            out.append(p)
    return [x for x in out if x]


def generate_production_plan(
    topic: str,
    video_type: str,
    dossier: dict,
    target_long_beats: int = 28,
    target_short_steps: int = 5,
    ckpt_dir: Path | None = None,
) -> dict:
    """Generate high-density, fact-checked Long-form (28+ beats) + Interactive Funnel Short (8 beats) with local checkpoints."""
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _cached_step(name: str, fn):
        if ckpt_dir is not None:
            fp = ckpt_dir / f"{name}.json"
            if fp.exists():
                print(f"    [ScriptCkpt] Reusing local checkpoint {fp.name}")
                return json.loads(fp.read_text(encoding="utf-8"))
        val = fn()
        if ckpt_dir is not None:
            (ckpt_dir / f"{name}.json").write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
        return val
    sources_rich = [
        {
            "source_id": s["source_id"],
            "publisher": s["publisher"],
            "domain": s["domain"],
            "title": s["title"],
            "published": s["published"],
            "metrics": s["extracted_metrics"][:10],
            "excerpt": s["summary"][:650],
        }
        for s in dossier.get("sources", [])[:8]
    ]
    valid_sids = [s["source_id"] for s in sources_rich] or ["SRC-01"]

    # Step 1: Master Narrative Outline + Thumbnail
    print("  [Scriptwriter V2] Step 1/3: Designing master 4-chapter narrative arc & spotlight thumbnail...")
    outline = _cached_step("00_outline", lambda: call_llm_json(
        "你是『Hey Digital (@hey_digitall)』科技頻道總編導。\n"
        + HEY_DIGITAL_FEWSHOT_ANCHOR
        + "\n請根據爬蟲抓到的公開全文與數據，規劃一支結構嚴謹、有深度反差洞察的長片大綱與縮圖標題。嚴格使用台灣繁體中文，絕不可顛倒高階款與入門款的價格高低（Pro 價格必然高於 Air / 標準版，頻寬單位必然是 GB/s 而非 MB/s）。只輸出 JSON。",
        json.dumps({
            "topic": topic,
            "video_type": video_type,
            "verified_metrics_pool": dossier.get("verified_metrics_pool", [])[:25],
            "sources": sources_rich,
            "schema": {
                "title_long": "具強烈痛點反差與好奇心的 Hey Digital 長片標題（22~36字）",
                "summary_paragraph": "140字內精華影片導讀（供資訊欄使用）",
                "product_a": "對比主角 A（例如 MacBook Air）",
                "product_b": "對比主角 B（例如 MacBook Pro）",
                "thumbnail": {
                    "line1": "縮圖上行超粗體標題（6~10字）",
                    "line2": "縮圖下行超粗體疑問句（6~10字，含問號）",
                    "gold_phrase": "line1 或 line2 內要標成香檳金的 3~5 字關鍵詞",
                    "image_prompt": "Minimalist dark Apple keynote studio spotlight shot of two sleek aluminum laptops side by side, silver and space black, subtle rim light, dark charcoal background, 8k commercial product photography, no text",
                },
                "chapters": [
                    {"num": "01", "title": "容量升級的價格陷阱", "en": "PRICING & TCO TRAP", "thesis": "入門版看似便宜，一旦補齊記憶體與硬碟，價差立刻縮小"},
                    {"num": "02", "title": "晶片頻寬與散熱天花板", "en": "BANDWIDTH & THERMALS", "thesis": "無風扇被動散熱與記憶體頻寬分級，決定了高強度工作與本機 AI 的持續表現"},
                    {"num": "03", "title": "螢幕光學與外接體驗", "en": "DISPLAY & WORKFLOW", "thesis": "120Hz ProMotion Mini-LED 峰值亮度與連接埠擴充性的真實日常體感差異"},
                    {"num": "04", "title": "三種預算終極買法", "en": "BUYING VERDICT", "thesis": "按真實使用情境切分，告訴誰買入門最香、誰直上高階才不後悔"},
                ],
            },
        }, ensure_ascii=False),
        temperature=0.38,
        max_tokens=1600,
    ))

    chapters = (outline.get("chapters") or [])[:4]
    prod_a = outline.get("product_a") or "標準輕薄款"
    prod_b = outline.get("product_b") or "Pro 進階款"

    # Step 2: Write 7 high-density beats per chapter (1 silent 1.8s chapter card + 6 narrated 6–9s visual scenes = 28 beats!)
    all_long_beats = []
    for ch_idx, ch in enumerate(chapters):
        ch_num = str(ch.get("num") or f"{ch_idx+1:02d}").zfill(2)
        print(f"  [Scriptwriter V2] Step 2/3: Writing Chapter {ch_num} ({ch.get('title')}) with 3D WebGL & B-Roll choreography...")

        # Silent musical 1.8s Chapter Opener Card (exact replica of ref_1TO7dz_data.mp4 00:00-00:01.8!)
        if ch_idx > 0:
            all_long_beats.append({
                "chapter_num": ch_num,
                "chapter_title": purify_tw_tech_text(ch.get("title", "")),
                "chapter_en": ch.get("en", "DEEP DIVE"),
                "chapter_sub": purify_tw_tech_text(ch.get("thesis", "")[:22]),
                "visual_type": "chapter_dark_card",
                "narration": "",  # Pure Rhodes electric-piano musical transition just like 1TO7dzrTuQg!
                "silent_duration": 1.85,
                "source_id": valid_sids[0],
            })

        prev_narrations = [b.get("narration", "") for b in all_long_beats if b.get("narration")]

        def _gen_chapter_beats(ch=ch, ch_num=ch_num, prev_list=prev_narrations):
            import difflib
            for attempt in range(3):
                res = call_llm_json(
                    "你是『Hey Digital (@hey_digitall)』王牌主筆與 Three.js 3D 視覺導演。\n"
                    + HEY_DIGITAL_FEWSHOT_ANCHOR
                    + f"\n現在專門撰寫第 {ch_num} 章《{ch.get('title')}》（核心論點：{ch.get('thesis')}），請輸出 **6 個連續、高資訊密度的全新敘事鏡次 (`beats`)**。\n"
                    "【絕對鐵律】：\n"
                    "1. 這 6 鏡必須 100% 聚焦於本章主題《" + str(ch.get("title")) + "》，嚴禁重複前面章節已經寫過的句子、開場白或相同數據（例如前面講過容量加價，後面章節就專心講散熱降頻、Mini-LED 1000 nits 螢幕亮度、外接埠或三種預算買法）！\n"
                    "2. 嚴禁照抄下方欄位說明文字！每個鏡次的 `narration` 約 42~62 個繁體中文字，語氣自然像真人在聊天分析。\n"
                    "3. 在 6 鏡中混合搭配最適合的 `visual_type`：\n"
                    "   - `\"cinema_broll_video\"`：硬體實拍動態畫面（填 `image_prompt` 純英文寫實硬體特寫，無文字）\n"
                    "   - `\"webgl_memory_lanes\"`：Three.js 3D 雙軌方塊對比（填 `hero_metric`, `hero_unit`, `secondary_metric`, `lanes: [{\"label\":...,\"value\":數字,\"display\":...}, ...]`）\n"
                    "   - `\"webgl_stacked_tco\"`：Three.js 3D 價格或規格堆疊柱（填 `hero_metric`, `hero_unit`, `stack_left_label`, `stack_left_layers: [{\"name\":...,\"val\":數字},...]`, `stack_right_label`, `stack_right_val`）\n"
                    "   - `\"webgl_ssd_particles\"`：Three.js 3D 儲存或資料流粒子飛入筆電（填 `hero_metric`, `hero_unit`）\n"
                    "   - `\"webgl_chip_die\"`：Three.js 3D 晶片封裝核心發光特寫（填 `chip_label`, `hero_metric`, `hero_unit`）\n"
                    "   - `\"webgl_dual_laptop\"`：Three.js 3D 雙機並排散熱熱力或速度對比（填 `hero_metric`, `hero_unit`, `left_label`, `right_label`, `mode`: `\"thermal\"` 或 `\"speed\"`）\n"
                    "   - `\"ui_configurator_crawl\"`：動態選配器 + 真實爬蟲報導螢光筆視窗（填 `source_id`, `highlight_quote` 摘錄本章相關 18~30 字重點結論）\n"
                    "只輸出合法 JSON。",
                    json.dumps({
                        "video_title": outline.get("title_long", topic),
                        "product_a": prod_a,
                        "product_b": prod_b,
                        "current_chapter": ch,
                        "forbidden_previous_sentences_do_not_repeat": prev_list[-12:],
                        "valid_source_ids": valid_sids,
                        "crawled_sources": sources_rich[:6],
                        "required_output_format": {
                            "beats": [
                                {
                                    "visual_type": "從上述 7 種選一（6 鏡請交錯不同類型）",
                                    "narration": f"請針對第 {ch_num} 章《{ch.get('title')}》撰寫全新口播內容，不可與前章重複",
                                    "source_id": valid_sids[ch_idx % len(valid_sids)],
                                    "hero_metric": "純數值如 1000 或 153",
                                    "hero_unit": "單位如 nits 或 GB/s",
                                }
                            ]
                        },
                    }, ensure_ascii=False),
                    temperature=0.45 + attempt * 0.12,
                    max_tokens=2800,
                )
                cand_beats = (res.get("beats") or [])[:6]
                has_dup = False
                seen_local = list(prev_list)
                for cb in cand_beats:
                    nt = cb.get("narration") or ""
                    if "請針對第" in nt or len(nt) < 16:
                        has_dup = True
                        break
                    for old_t in seen_local:
                        if difflib.SequenceMatcher(None, nt, old_t).ratio() > 0.50:
                            has_dup = True
                            break
                    seen_local.append(nt)
                if cand_beats and not has_dup:
                    return res
                print(f"    [ScriptQA] Chapter {ch_num} had overlap with previous text (attempt {attempt+1}); regenerating fresh prose...")
            return res

        ch_res = _cached_step(f"01_ch_{ch_num}", _gen_chapter_beats)
        ch_beats = (ch_res.get("beats") or [])[:6]
        for b in ch_beats:
            b["chapter_num"] = ch_num
            b["chapter_title"] = purify_tw_tech_text(ch.get("title", "深度解析"))
            b["chapter_en"] = ch.get("en", "DEEP DIVE")
            all_long_beats.append(b)

    # Step 3: Write High-Density 8-Beat Interactive Traffic-Funnel Short (~65–80s, 1:1 to QrBoNy0wse8 choreography!)
    print("  [Scriptwriter V2] Step 3/3: Designing 8-beat Interactive Traffic-Funnel Short (Pop-in Stickers + Animated Toggle/Touch UI)...")
    short_plan = _cached_step("02_short_plan", lambda: call_llm_json(
        "你是『Hey Digital (@hey_digitall)』YouTube Shorts 導流短片導演。\n"
        "請針對剛完成的長片，設計一支節奏明快、資訊量飽滿、總長約 65~80 秒的 **8 鏡直式導流短片 (`short_beats`)**：\n"
        "- 第 1 鏡 (`visual_type: \"short_hook_dynamic\"`)：黃金前 10~13 秒強痛點開場（約 45~60 字口播），提供 **5 張依序隨語音彈出的動態貼紙** `pop_stickers: [{\"text\":\"...\",\"style\":\"white|yellow|green\"}, ...]`（每張 4~7 字，如原片『買 Air 還是 Pro？』、『差價陷阱！』、『不是只看跑分』、『3 個隱藏刀法』、『看完省下冤枉錢 →』）！\n"
        "- 第 2~6 鏡（共 5 個核心實戰步驟，`visual_type: \"short_phone_interactive\"`）：每鏡約 38~52 字自然口播，並在全尺寸 iPhone 螢幕內動態展示 **4 列互動規格／選購開關卡片 `ui_rows`**（含自動滑鼠點擊波紋與 iOS 綠色開關逐一開啟效果！每列含 `title`, `subtitle`, `badge`, `toggle_on`: true/false），以及頂部 `step_title`(4~6字)、`flow_steps`(3個短詞)、側邊彈出黃色刀法貼紙 `callout_tag`(4~6字) 與 `source_id`！\n"
        "- 第 7 鏡 (`visual_type: \"short_broll_summary\"`)：實拍硬體動態展示收束（約 30~40 字口播，給出一句話選購口訣）。\n"
        "- 第 8 鏡 (`visual_type: \"cta_funnel_endcard\"`)：明確口播導流：「這支短片只是精華重點，我們在完整長片裡實測了完整頻寬、散熱曲線與全部公開數據來源，現在就點擊下方關聯影片收看完整解析！」\n"
        "嚴格使用台灣繁體中文，只輸出合法 JSON。",
        json.dumps({
            "parent_long_title": outline.get("title_long", topic),
            "product_a": prod_a,
            "product_b": prod_b,
            "valid_source_ids": valid_sids,
            "sources": sources_rich[:5],
            "schema_example": {
                "title_short": "買輕薄款還是直上 Pro？5 個關鍵刀法一次看懂！ #Shorts",
                "short_beats": [
                    {
                        "visual_type": "short_hook_dynamic",
                        "topic_pill": "筆電選購避坑",
                        "pop_stickers": [
                            {"text": "買 Air 還是 Pro？", "style": "white"},
                            {"text": "升級完差不多貴？", "style": "white"},
                            {"text": "別只看跑分數據", "style": "yellow"},
                            {"text": "5 個關鍵刀法", "style": "green"},
                            {"text": "看完再下單 →", "style": "yellow"},
                        ],
                        "narration": "明明帳面上入門款便宜了好幾百美元，為什麼一加完記憶體和硬碟，價格馬上就貼近高階款？今天用一分鐘把最核心的五個硬體刀法一次講清楚，建議先收藏起來對照選配！",
                        "source_id": "SRC-01",
                        "image_prompt": "Vertical overhead studio shot of hands opening sleek space black aluminum MacBook laptop on wooden desk, moody rim lighting, no text",
                    },
                    {
                        "visual_type": "short_phone_interactive",
                        "topic_pill": "筆電選購避坑",
                        "step_title": "真實價差帳",
                        "flow_steps": ["看起步容量", "算升級加價", "比同規總價"],
                        "callout_tag": "價差縮小",
                        "ui_rows": [
                            {"title": "入門基礎配置", "subtitle": "16GB + 512GB 日常文書夠用", "badge": "省預算首選", "toggle_on": True},
                            {"title": "加價升到 24GB + 1TB", "subtitle": "升級費瞬間吃掉價差優勢", "badge": "+$400", "toggle_on": False},
                            {"title": "直上 Pro 同級容量", "subtitle": "多換來主動散熱與高刷螢幕", "badge": "性價比反超", "toggle_on": True},
                            {"title": "選購鐵律：買 Air 不升頂", "subtitle": "一旦加超過兩階就直接上 Pro", "badge": "核心結論", "toggle_on": True},
                        ],
                        "narration": "第一步先算真實總價：入門款不加配最划算，但只要你同時升級記憶體和硬碟，加價後的金額幾乎已經摸到高階款門檻，這時候直上反而換來更多硬體規格。",
                        "source_id": "SRC-01",
                        "image_prompt": "Vertical close up of sleek laptop keyboard and trackpad, studio lighting, no text",
                    }
                ],
            },
        }, ensure_ascii=False),
        temperature=0.4,
        max_tokens=3000,
    ))

    # Normalize & link verifiable crawled sources + generate 100% digit-free `tts_text`!
    src_map = {s["source_id"]: s for s in dossier.get("sources", [])}
    fallback_src = dossier["sources"][0] if dossier.get("sources") else {
        "source_id": "SRC-01",
        "publisher": "Public Tech Spec",
        "domain": "wikipedia.org",
        "title": topic,
        "url": "https://en.wikipedia.org/wiki/Apple_silicon",
        "published": "2026",
        "summary": topic,
        "extracted_metrics": [],
    }
    used_sources = {}
    accents = ["#0E6EB8", "#B84A14", "#0E7C47", "#5B50D6"]

    for idx, b in enumerate(all_long_beats, start=1):
        b["id"] = f"L{idx:02d}"
        vtype = b.get("visual_type") or ""
        if vtype != "chapter_dark_card" and vtype not in LONG_SCENE_TYPES:
            b["visual_type"] = LONG_SCENE_TYPES[idx % len(LONG_SCENE_TYPES)]
        b["accent_hex"] = b.get("accent_hex") or accents[idx % len(accents)]
        b["narration"] = purify_tw_tech_text(b.get("narration") or "")
        b["tts_text"] = verbalize_for_mandarin_tts(b["narration"]) if b["narration"] else ""
        b["captions"] = split_into_captions(b["narration"], max_len=18) if b["narration"] else []
        if not b.get("image_prompt"):
            b["image_prompt"] = "Photorealistic commercial studio shot of sleek aluminum laptops and glowing silicon chip on walnut slat desk, 35mm cinema lens, no text"
        sid = b.get("source_id") if b.get("source_id") in src_map else valid_sids[(idx - 1) % len(valid_sids)]
        b["source_id"] = sid
        s_obj = src_map.get(sid, fallback_src)
        used_sources[sid] = s_obj
        b["source_detail"] = s_obj
        dom = s_obj.get("domain", "")
        tag_sub = s_obj.get("published", "")[:10] if "news.google" in dom else dom
        b["source_badge"] = f"資料來源：{s_obj['publisher']} · {tag_sub}"

    short_beats = short_plan.get("short_beats") or []
    if short_beats:
        short_beats[0]["visual_type"] = "short_hook_dynamic"
        if short_beats[-1].get("visual_type") != "cta_funnel_endcard":
            short_beats.append({
                "visual_type": "cta_funnel_endcard",
                "narration": "這支短片只是重點濃縮，我們在完整長片裡實測了完整頻寬、散熱表現與全部公開數據來源，現在就點擊下方關聯影片收看完整解析！",
                "source_id": valid_sids[0],
            })
    for idx, sb in enumerate(short_beats, start=1):
        sb["id"] = f"S{idx:02d}"
        if 1 < idx < len(short_beats) and sb.get("visual_type") not in ("short_phone_interactive", "short_broll_summary"):
            sb["visual_type"] = "short_phone_interactive"
        sb["narration"] = purify_tw_tech_text(sb.get("narration") or "")
        sb["tts_text"] = verbalize_for_mandarin_tts(sb["narration"])
        sb["captions"] = split_into_captions(sb["narration"], max_len=14)
        sid = sb.get("source_id") if sb.get("source_id") in src_map else valid_sids[(idx - 1) % len(valid_sids)]
        sb["source_id"] = sid
        s_obj = src_map.get(sid, fallback_src)
        used_sources[sid] = s_obj
        sb["source_detail"] = s_obj
        if not sb.get("image_prompt"):
            sb["image_prompt"] = "Vertical studio close-up of sleek modern aluminum laptop and glowing silicon chip on walnut desk, warm rim lighting, no text"

    for s in dossier.get("sources", [])[:8]:
        used_sources.setdefault(s["source_id"], s)

    th = outline.get("thumbnail") or {}
    th["line1"] = purify_tw_tech_text(th.get("line1") or f"{prod_a} 對決 {prod_b}")
    th["line2"] = purify_tw_tech_text(th.get("line2") or "實測後買誰才不虧？")
    th["gold_phrase"] = purify_tw_tech_text(th.get("gold_phrase") or "")

    return {
        "topic": purify_tw_tech_text(topic),
        "video_type": video_type,
        "title_long": purify_tw_tech_text(outline.get("title_long") or topic),
        "title_short": purify_tw_tech_text(short_plan.get("title_short") or f"{topic[:18]} #Shorts"),
        "summary_paragraph": purify_tw_tech_text(outline.get("summary_paragraph") or topic),
        "product_a": purify_tw_tech_text(prod_a),
        "product_b": purify_tw_tech_text(prod_b),
        "thumbnail": th,
        "long_beats": all_long_beats,
        "short_beats": short_beats,
        "cited_sources": list(used_sources.values()),
    }
