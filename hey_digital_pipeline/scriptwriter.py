#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon — Crawled-Fact AI Script & Storyboard Generator (Local Master)
==========================================================================================
Generates reusable, multi-genre Hey Digital scripts + visual beat plans (Long-form +
Traffic-Funnel Short + Thumbnail + Cited YouTube Descriptions) grounded in live crawled
public sources (`[SRC-01]..[SRC-12]`).

Supported video types (`--video-type`):
- `versus_guide`   : 雙機/多機橫評選購對決（如 Air vs Pro、平板 vs 筆電）
- `ai_benchmark`   : 本機 AI 算力、統一記憶體與頻寬實測避坑
- `deep_review`    : 單品深度評測（硬體亮點 vs 真實妥協痛點）
- `hidden_tips`    : 新機必做設定與進階隱藏技巧
- `news_analysis`  : 科技新品發布與產業刀法深度拆解

LLM Backends (auto-cascading):
1. Gemini API (`GEMINI_API_KEY` / `GOOGLE_API_KEY`)
2. DeepSeek / OpenAI-compatible API (`DEEPSEEK_API_KEY` / `OPENAI_API_KEY`)
3. Remote RTX Pro 6000 PyTorch LLM fallback
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request
from pathlib import Path
from typing import Optional


VIDEO_TYPE_PROFILES = {
    "versus_guide": "雙機或多陣營深度選購對決（拆解帳面參數迷思、真實使用場景摩擦、隱藏成本經濟帳、明確購買分流建議）",
    "ai_benchmark": "本機 AI 部署與硬體算力實測（拆解記憶體容量 vs 頻寬天花板、Prefill 與 Decode 差異、模型梯隊實測與選購建議）",
    "deep_review": "旗艦科技產品深度評測（工業設計與面板光學進化、晶片刀法與散熱續航取捨、生態定位與目標受眾）",
    "hidden_tips": "必做關鍵設定與效率實戰指南（打破預設耗電或畫質陷阱、逐步實操路徑、前後體感差異對比）",
    "news_analysis": "最新科技硬體與趨勢深度拆解（從公開財報/跑分/供應鏈規格看懂廠商定價邏輯與刀法真相）",
}


def _extract_json_block(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if m:
        text = m.group(1)
    else:
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and j > i:
            text = text[i : j + 1]
    return json.loads(text)


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.45) -> dict:
    ctx = ssl.create_default_context()

    # 1. Gemini API if configured
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        model = os.environ.get("HEYDIGITAL_GEMINI_MODEL", "gemini-2.5-pro")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        return _extract_json_block(raw)

    # 2. OpenAI / DeepSeek API
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        is_ds = bool(os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"))
        base = os.environ.get("HEYDIGITAL_LLM_BASE_URL") or ("https://api.deepseek.com/v1" if is_ds else "https://api.openai.com/v1")
        model = os.environ.get("HEYDIGITAL_LLM_MODEL") or ("deepseek-chat" if is_ds else "gpt-4.1")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": 8000,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{base.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        raw = data["choices"][0]["message"]["content"]
        return _extract_json_block(raw)

    raise RuntimeError("No GEMINI_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY found in environment.")


def formulate_topic_from_candidates(candidates_report: dict, existing_titles: list[str], video_type: str = "ai_benchmark") -> dict:
    """Autonomously select and frame a high-CTR Hey Digital topic from live crawled tech feeds."""
    sys_p = (
        "你是 YouTube 科技深度頻道『Hey Digital (@hey_digitall)』的總編導 Agent。"
        "頻道風格：台灣繁體中文、理性硬核但不掉書袋、擅長拆解 Apple / AI PC / GPU / 行動裝置的底層規格刀法（如記憶體頻寬、OLED、散熱、TCO 購買帳），"
        "標題極具張力與選購痛點反差。請根據最新爬蟲候選新聞，自動選定一個最適合本期更新的主題，並給出精準搜尋關鍵字。"
        "嚴格輸出 JSON。"
    )
    usr_p = json.dumps({
        "target_video_type": video_type,
        "video_type_desc": VIDEO_TYPE_PROFILES.get(video_type, ""),
        "already_published_titles_avoid_exact_duplicate": existing_titles[-12:],
        "live_crawled_candidates": candidates_report.get("top_candidates", [])[:8],
        "required_json_schema": {
            "selected_topic": "繁體中文核心主題（具選購對比或痛點張力）",
            "video_type": video_type,
            "angle_rationale": "為何此主題契合 Hey Digital 受眾且與舊片互補",
            "search_queries": ["英文或繁中精準爬蟲檢索詞1", "檢索詞2", "檢索詞3", "檢索詞4"],
            "wiki_terms": ["Wikipedia 條目詞1", "Wikipedia 條目詞2"],
        },
    }, ensure_ascii=False, indent=2)
    return call_llm_json(sys_p, usr_p, temperature=0.4)


def split_into_captions(narration: str, max_len: int = 19) -> list[str]:
    """Split narration into clean single-line Hey Digital captions (no trailing punctuation)."""
    parts = re.split(r"[，。！？；：、\n]+", narration.strip())
    out = []
    for p in parts:
        p = p.strip(" 「」『』\"'，。！？；：")
        if not p:
            continue
        # Ensure space around ASCII words/numbers inside CJK
        p = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", p)
        p = re.sub(r"([A-Za-z0-9%])([\u4e00-\u9fff])", r"\1 \2", p)
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) <= max_len:
            out.append(p)
        else:
            # Split at middle space or midpoint
            mid = len(p) // 2
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
    target_long_beats: int = 22,
    target_short_steps: int = 5,
) -> dict:
    """Generate full cited Long-form + Traffic-Funnel Short script & visual storyboard."""
    sources_compact = [
        {
            "source_id": s["source_id"],
            "publisher": s["publisher"],
            "domain": s["domain"],
            "title": s["title"],
            "url": s["url"],
            "published": s["published"],
            "summary": s["summary"][:380],
            "metrics": s["extracted_metrics"],
        }
        for s in dossier.get("sources", [])[:10]
    ]

    sys_prompt = """你是『Hey Digital (@hey_digitall)』YouTube 頻道的王牌科技主筆與視覺分鏡導演。
你的任務是將「即時爬蟲擷取的公開資訊與來源 (`sources`)」深度融合進一支完整長片腳本，以及一支對應導流（推流）的 YouTube Short 短片腳本中。

### 嚴格遵守 Hey Digital 頻道風格規範：
1. **語言與口吻**：台灣繁體中文（用語如：記憶體、頻寬、晶片、硬碟、螢幕、筆電、演算法、性價比）。第一人稱「我們」或直接對話「你」，語氣冷靜、犀利、有洞察力，不喊口號，直接拆解廠商刀法與真實使用體感。
2. **公開爬蟲資訊融合與來源標註**：
   - 必須將 `sources` 中的真實數據、規格、定價、媒體實測或社群討論自然寫進旁白，並在每個引用公開資訊的鏡頭填入對應的 `source_id`（例如 `"SRC-01"`）與簡短螢幕來源標籤 `source_badge`（例如 `"資料來源：Ars Technica / Wikipedia 公開規格"`）。
   - 嚴禁捏造不存在的 `source_id`；只能引用輸入提供的 `SRC-01` ~ `SRC-10`。
3. **長片結構 (`long_beats`，共約 20~24 個鏡次，分為 5 個章節 `01`~`05`)**：
   - 每個新章節的第一個鏡次設為 `visual_type: "chapter_dark_card"`（約 1 句簡潔破題旁白，15~28 字）。
   - 其餘鏡次交替使用：
     - `"studio_isometric_infographic"`（佔約 45%）：暖白棚拍 3D 等距數據對比圖卡！必須提供超大指標 `hero_metric`（如 `"256"`）、單位 `hero_unit`（如 `"GB/s"`）、主題色 `accent_hex`（`"#B84A14"` 焦糖橘 / `"#0E7C47"` 翡翠綠 / `"#0E6EB8"` 鈷藍 / `"#5B50D6"` 靛紫），以及 2~4 組具體對比數據條 `comparison_bars: [{"label": "...", "value": 85, "display": "..."}]`。
     - `"cinema_tech_broll"`（佔約 40%）：由 RTX Pro 6000 生成的高質感科技攝影棚實景圖！必須提供純英文、無文字浮水印的高畫質商業硬體攝影 Prompt (`image_prompt`，描述胡桃木格柵背景、柔和輪廓光、微距晶片或鋁合金筆電/主機特寫，明確寫 `no text, no letters, clean studio photography, 8k, 35mm lens`)，並可附上 `overlay_news_card`（展示爬蟲抓到的媒體標題摘要浮窗）。
   - 每個一般鏡次的 `narration` 長度約 45~75 個中文字（朗讀約 10~16 秒），總長約 4~5.5 分鐘精華深度長片。
4. **對應導流 Short 結構 (`short_beats`，共 7 個鏡次，總長約 65~85 秒)**：
   - Short 必須是這支長片的「精準推流預告精華」！
   - 第 1 鏡：`visual_type: "hook_sticker_broll"`，3~5 秒強痛點開場，附 2 個短貼紙標語 `sticker_top`, `sticker_bottom`（各 4~8 字）與英文直式硬體攝影 `image_prompt`。
   - 第 2~6 鏡（共 5 步）：`visual_type: "step_phone_mockup"`，提供 `step_index` (`1`~`5`), `step_total` (`5`), `step_title`（4~7 字核心結論）, `flow_steps`（3 個短步驟詞陣列 `["...", "...", "..."]`）, `callout_tag`（4~6 字黃底貼紙詞）, `card_metric`, `card_points`（2~3 條數據重點）, `source_id`, 以及旁白 `narration`（約 30~45 字）。
   - 第 7 鏡：`visual_type: "cta_funnel_endcard"`，明確口播引導觀眾點擊下方關聯長片收看完整深度評測與全部數據來源！
5. **縮圖 (`thumbnail`)**：
   - `line1`（上行 6~10 字）、`line2`（下行 6~10 字）、`gold_phrase`（必須完全包含在 `line1` 或 `line2` 內的關鍵高亮詞，將渲染為香檳金 `#F0CA86`），以及英文暗黑聚光燈產品主體 `image_prompt`。

請嚴格回傳合法 JSON 物件。"""

    user_prompt = json.dumps({
        "topic": topic,
        "video_type": video_type,
        "video_type_guide": VIDEO_TYPE_PROFILES.get(video_type, ""),
        "target_long_beats": target_long_beats,
        "target_short_steps": target_short_steps,
        "crawled_public_sources": sources_compact,
        "output_json_example_skeleton": {
            "title_long": "...",
            "title_short": "... #Shorts",
            "summary_paragraph": "...",
            "thumbnail": {
                "line1": "...",
                "line2": "...",
                "gold_phrase": "...",
                "image_prompt": "Minimalist dark studio spotlight product shot of ..., dramatic rim lighting, matte black background, ultra detailed industrial design, no text, 8k",
            },
            "chapters": [
                {"num": "01", "title": "...", "en": "OPENING DILEMMA"}
            ],
            "long_beats": [
                {
                    "id": "L01",
                    "chapter_num": "01",
                    "chapter_title": "...",
                    "chapter_en": "OPENING DILEMMA",
                    "visual_type": "chapter_dark_card",
                    "narration": "...",
                    "source_id": "SRC-01",
                    "source_badge": "資料來源：...",
                    "hero_metric": "",
                    "hero_unit": "",
                    "accent_hex": "#0E6EB8",
                    "comparison_bars": [],
                    "image_prompt": "Sleek aluminum laptop and compact AI workstation on warm walnut wood slat desk, moody studio lighting, shallow depth of field, 35mm cinema photo, no text",
                    "overlay_news_card": True,
                }
            ],
            "short_beats": [
                {
                    "id": "S01",
                    "visual_type": "hook_sticker_broll",
                    "topic_pill": "...",
                    "sticker_top": "...",
                    "sticker_bottom": "...",
                    "narration": "...",
                    "source_id": "SRC-01",
                    "image_prompt": "Close up vertical shot of sleek modern laptop chip glowing subtly on dark desk, studio lighting, no text",
                }
            ],
        },
    }, ensure_ascii=False, indent=2)

    plan = call_llm_json(sys_prompt, user_prompt, temperature=0.45)

    # Post-process captions & bind full source URLs for verifiable citations
    src_map = {s["source_id"]: s for s in dossier.get("sources", [])}
    used_sources = {}

    for idx, b in enumerate(plan.get("long_beats", []), start=1):
        b["id"] = f"L{idx:02d}"
        b["captions"] = split_into_captions(b.get("narration", ""), max_len=20)
        sid = b.get("source_id") or ""
        if sid in src_map:
            used_sources[sid] = src_map[sid]
            b["source_detail"] = src_map[sid]
            if not b.get("source_badge"):
                b["source_badge"] = f"資料來源：{src_map[sid]['publisher']} ({src_map[sid]['domain']})"

    for idx, sb in enumerate(plan.get("short_beats", []), start=1):
        sb["id"] = f"S{idx:02d}"
        sb["captions"] = split_into_captions(sb.get("narration", ""), max_len=15)
        sid = sb.get("source_id") or ""
        if sid in src_map:
            used_sources[sid] = src_map[sid]
            sb["source_detail"] = src_map[sid]

    # Ensure at least top crawled sources are attached
    if not used_sources:
        for s in dossier.get("sources", [])[:5]:
            used_sources[s["source_id"]] = s

    plan["topic"] = topic
    plan["video_type"] = video_type
    plan["cited_sources"] = list(used_sources.values())
    return plan
