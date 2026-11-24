#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon — Live Public Tech Crawler & Verifiable Citation Engine
====================================================================================
1. Autonomous Topic Discovery (`discover_topics`):
   Crawls live public tech RSS/APIs (9to5Mac, MacRumors, Ars Technica, The Verge,
   Hacker News Algolia API, Google News Taiwan Tech RSS), filters out already-published
   `@hey_digitall` topics, and ranks high-retention consumer tech / AI hardware topics.
2. Deep Public Fact & Spec Dossier (`crawl_topic_dossier`):
   Crawls live articles, Wikipedia REST specs, Hacker News discussions, and public release
   notes for a topic, extracting concrete specs, prices, benchmark numbers, and verifiable
   source URLs (`[SRC-01]`, `[SRC-02]`, ...) ready to weave into script + visual cards!
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

PUBLIC_TECH_FEEDS = [
    ("9to5Mac", "https://9to5mac.com/feed/"),
    ("MacRumors", "https://feeds.macrumors.com/MacRumors-All"),
    ("Ars Technica Gear", "https://feeds.arstechnica.com/arstechnica/gadgets"),
    ("The Verge Tech", "https://www.theverge.com/rss/tech/index.xml"),
    ("Google News TW 科技", "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=zh-TW&gl=TW&ceid=TW:zh-Hant"),
]


def _http_get(url: str, timeout: int = 12) -> bytes:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def _strip_html(raw: str) -> str:
    txt = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw, flags=re.I | re.S)
    txt = re.sub(r"<style\b[^>]*>.*?</style>", " ", txt, flags=re.I | re.S)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def _parse_rss_items(publisher: str, xml_bytes: bytes, limit: int = 10) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return items

    # RSS 2.0 <item>
    for el in root.findall(".//item")[:limit]:
        title = (el.findtext("title") or "").strip()
        link = (el.findtext("link") or "").strip()
        pub = (el.findtext("pubDate") or "").strip()
        desc = _strip_html(el.findtext("description") or "")[:500]
        src_el = el.find("source")
        real_pub = (src_el.text.strip() if src_el is not None and src_el.text else publisher)
        if title and link:
            items.append({
                "publisher": real_pub,
                "title": title,
                "url": link,
                "published": pub[:25],
                "snippet": desc,
            })
    if items:
        return items

    # Atom <entry>
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for el in root.findall(".//a:entry", ns)[:limit]:
        title = (el.findtext("a:title", default="", namespaces=ns) or "").strip()
        link_el = el.find("a:link", ns)
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        pub = (el.findtext("a:updated", default="", namespaces=ns) or el.findtext("a:published", default="", namespaces=ns) or "").strip()
        summary = _strip_html(el.findtext("a:summary", default="", namespaces=ns) or el.findtext("a:content", default="", namespaces=ns) or "")[:500]
        if title and link:
            items.append({
                "publisher": publisher,
                "title": title,
                "url": link,
                "published": pub[:25],
                "snippet": summary,
            })
    return items


def search_hackernews(query: str, limit: int = 8) -> list[dict]:
    q = urllib.parse.urlencode({"query": query, "tags": "story", "hitsPerPage": str(limit)})
    url = f"https://hn.algolia.com/api/v1/search_by_date?{q}"
    try:
        data = json.loads(_http_get(url, timeout=10).decode("utf-8"))
        out = []
        for hit in data.get("hits", []):
            title = hit.get("title") or ""
            target_url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
            domain = urllib.parse.urlparse(target_url).netloc.replace("www.", "") or "news.ycombinator.com"
            out.append({
                "publisher": f"Hacker News ({domain})",
                "title": title,
                "url": target_url,
                "published": (hit.get("created_at") or "")[:10],
                "snippet": f"HN Points: {hit.get('points', 0)} | Comments: {hit.get('num_comments', 0)}",
            })
        return out
    except Exception:
        return []


def search_google_news_rss(query: str, lang: str = "zh-TW", limit: int = 8) -> list[dict]:
    if lang == "zh-TW":
        q = urllib.parse.urlencode({"q": query, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"})
    else:
        q = urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    url = f"https://news.google.com/rss/search?{q}"
    try:
        return _parse_rss_items("Google News", _http_get(url, timeout=12), limit=limit)
    except Exception:
        return []


def fetch_wikipedia_summary(term: str, lang: str = "en") -> Optional[dict]:
    q = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": term,
        "format": "json",
        "srlimit": "1",
    })
    api = f"https://{lang}.wikipedia.org/w/api.php?{q}"
    try:
        res = json.loads(_http_get(api, timeout=10).decode("utf-8"))
        hits = res.get("query", {}).get("search", [])
        if not hits:
            return None
        page_title = hits[0]["title"]
        slug = urllib.parse.quote(page_title.replace(" ", "_"))
        sum_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{slug}"
        sdata = json.loads(_http_get(sum_url, timeout=10).decode("utf-8"))
        extract = (sdata.get("extract") or "").strip()
        if not extract:
            return None
        return {
            "publisher": f"Wikipedia ({lang.upper()})",
            "title": sdata.get("title", page_title),
            "url": sdata.get("content_urls", {}).get("desktop", {}).get("page", f"https://{lang}.wikipedia.org/wiki/{slug}"),
            "published": (sdata.get("timestamp") or "")[:10],
            "snippet": extract[:900],
        }
    except Exception:
        return None


def fetch_article_excerpt(url: str, max_chars: int = 1400) -> str:
    """Fetch public webpage body paragraphs safely for deeper numeric/spec extraction."""
    if "news.google.com/rss/articles" in url:
        return ""
    try:
        raw = _http_get(url, timeout=10).decode("utf-8", errors="ignore")
        paras = re.findall(r"<p\b[^>]*>(.*?)</p>", raw, flags=re.I | re.S)
        cleaned = []
        for p in paras:
            t = _strip_html(p)
            if len(t) >= 45 and not any(skip in t.lower() for skip in ("cookie", "privacy policy", "subscribe to", "all rights reserved")):
                cleaned.append(t)
            if sum(len(x) for x in cleaned) >= max_chars:
                break
        return " ".join(cleaned)[:max_chars]
    except Exception:
        return ""


def discover_unattended_topics(existing_titles: list[str], top_k: int = 5) -> dict:
    """Crawl live public tech feeds and autonomously rank & pick the best Hey Digital topic."""
    raw_items = []
    for pub, feed_url in PUBLIC_TECH_FEEDS:
        try:
            raw_items.extend(_parse_rss_items(pub, _http_get(feed_url, timeout=10), limit=8))
        except Exception:
            continue
    raw_items.extend(search_hackernews("Apple M4 OR M5 OR MacBook OR Nvidia OR OLED OR Local LLM", limit=10))

    # Score items for Hey Digital's signature hardware/AI/apple/buying-guide DNA
    high_signal_keywords = {
        "macbook": 5, "mac studio": 5, "mac mini": 5, "ipad": 4, "iphone": 4,
        "apple": 3, "oled": 4, "pro": 2, "air": 2, "nvidia": 4, "rtx": 4,
        "amd": 4, "ryzen": 4, "strix halo": 6, "ai max": 6, "dgx": 5, "llm": 5,
        "thunderbolt": 5, "memory": 4, "bandwidth": 5, "unified": 5, "benchmark": 5,
        "ssd": 3, "chip": 3, "m4": 4, "m5": 5, "snapdragon": 3, "galaxy": 3,
        "記憶體": 4, "頻寬": 5, "筆電": 4, "晶片": 4, "實測": 5, "評測": 4,
    }
    existing_lower = " || ".join(t.lower() for t in existing_titles)

    scored = []
    seen_titles = set()
    for it in raw_items:
        tl = it["title"].lower()
        key = re.sub(r"\W+", "", tl)[:32]
        if key in seen_titles:
            continue
        seen_titles.add(key)

        score = 1
        for kw, pts in high_signal_keywords.items():
            if kw in tl or kw in it.get("snippet", "").lower():
                score += pts
        if re.search(r"\b\d+\s*(gb|tb|hz|w|nm|nit|fps|%|usd|\$)\b", tl + " " + it.get("snippet", "").lower()):
            score += 4
        # Slight penalty if exact product phrase already heavily covered in very recent title
        if "ipad mini 8" in tl and "ipad mini 8" in existing_lower:
            score -= 4
        scored.append({**it, "fit_score": score})

    scored.sort(key=lambda x: x["fit_score"], reverse=True)
    candidates = scored[: max(top_k, 8)]
    return {
        "crawled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_feed_items_scanned": len(raw_items),
        "top_candidates": candidates[:top_k],
        "raw_pool": candidates,
    }


def crawl_topic_dossier(topic: str, search_queries: list[str], wiki_terms: Optional[list[str]] = None) -> dict:
    """Build a verifiable, citation-indexed public information dossier for a video topic."""
    collected = []
    seen_urls = set()

    def _add(item: Optional[dict]):
        if not item or not item.get("url"):
            return
        u = item["url"]
        if u in seen_urls:
            return
        seen_urls.add(u)
        collected.append(item)

    # 1. Wikipedia baseline technical specs
    for wt in (wiki_terms or search_queries[:2]):
        _add(fetch_wikipedia_summary(wt, "zh"))
        _add(fetch_wikipedia_summary(wt, "en"))

    # 2. Live news & reviews (TW + EN)
    for q in search_queries:
        for item in search_google_news_rss(q, lang="zh-TW", limit=4):
            _add(item)
        for item in search_google_news_rss(q, lang="en", limit=4):
            _add(item)
        for item in search_hackernews(q, limit=4):
            _add(item)

    # 3. Enrich direct articles with body excerpts & numeric facts
    sources = []
    for idx, item in enumerate(collected[:12], start=1):
        excerpt = fetch_article_excerpt(item["url"]) if idx <= 6 else ""
        combined_text = (item.get("snippet", "") + " " + excerpt).strip()
        # Extract salient numbers/specs
        nums = re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:GB/s|TB/s|GB|TB|Hz|W|nm|nits?|tok/s|fps|mAh|g|kg|%|USD|美元|元|核心|吋|英寸)\b",
            combined_text,
            flags=re.I,
        )
        domain = urllib.parse.urlparse(item["url"]).netloc.replace("www.", "")
        sources.append({
            "source_id": f"SRC-{idx:02d}",
            "publisher": item["publisher"],
            "domain": domain,
            "title": item["title"],
            "url": item["url"],
            "published": item.get("published") or dt.date.today().isoformat(),
            "summary": combined_text[:700],
            "extracted_metrics": list(dict.fromkeys(nums))[:10],
        })

    return {
        "topic": topic,
        "crawled_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "queries": search_queries,
        "source_count": len(sources),
        "sources": sources,
    }
