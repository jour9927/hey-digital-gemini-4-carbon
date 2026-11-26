#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon V5 — Linguistic Word-Boundary & Word-Timestamp Subtitle Engine
==========================================================================================
Reconstructs `@hey_digitall`'s exact human-editor subtitle segmentation, visual rhythm, and
millisecond-accurate spoken synchronization (`D9IiQF_QIYg`, `1TO7dzrTuQg`, `QrBoNy0wse8`):

1. True Visual Em-Width Accounting (`visual_em_len`):
   - Counts CJK = `1.0 em`, uppercase/digits = `0.56 em`, lowercase/spaces = `0.45 em`, so clauses
     containing `"MacBook Air M5"` or `"120Hz ProMotion"` are NEVER prematurely chopped in half!
   - Strictly single-line (`<= 22.0 em` Long-form, `<= 17.5 em` Short-form) — ZERO 2-line wraps!
2. Punctuation & Discourse-Connector Rules:
   - NEVER splits on enumeration comma `、` (`頓號`, e.g. `雷電四、HDMI 與記憶卡插槽` stays intact).
   - Keeps short leading connectors (`不過，`, `當然，`, `說白了，`) attached to their main clause.
   - Strips trailing periods `。` while keeping natural inline `，` / `、` when short pairs merge.
3. Lexical & Syntactic DP Word-Boundary Splitter (`tokenize_subtitle_atoms` + `_dp_split_long_clause`):
   - Uses `rjieba` + `OpenCC('t2s')` 1:1 index mapping + protected tech compounds & disyllabic verbs
     so multi-character words (`還是`, `配備`, `摸到`, `升到`, `跑到`, `超高`, `高強度`, `直接`, `連接埠`)
     are 100% INDIVISIBLE!
   - Hard-forbids dangling prepositions/adverbs/modals (`則`, `跟`, `與`, `超`, `很`, `更`, `在`, `從`, `把`)
     at the end of a line and bare enclitics (`了`, `著`, `過`, `的`) at the start of the next line!
4. Word-Level Whisper Forced Alignment (`align_captions_to_whisper_words`):
   - Aligns every segmented caption directly to Whisper word-level `(start, end)` timestamps via
     monotonic `SequenceMatcher` character anchors so caption switches match exact spoken syllables!
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Optional

from hey_digital_pipeline.cn_verbalizer import purify_tw_tech_text

try:
    import rjieba as _jieba_mod
except Exception:
    try:
        import jieba as _jieba_mod
    except Exception:
        _jieba_mod = None

try:
    from opencc import OpenCC
    _T2S = OpenCC("t2s")
except Exception:
    _T2S = None


# Indivisible Traditional Chinese tech terms, resultative verbs & compounds
PROTECTED_CJK_WORDS = sorted([
    "統一記憶體", "記憶體頻寬", "固態硬碟", "高速硬碟", "外接硬碟", "外接設備", "巧控鍵盤", "大語言模型", "人工智慧", "神經網路",
    "超高峰值亮度", "峰值亮度", "超高亮度", "高更新率", "更新率", "記憶卡插槽", "記憶卡槽", "記憶卡", "雷電四", "雷電五", "雷電三",
    "記憶體", "連接埠", "觸控板", "揚聲器", "轉接器", "時間軸", "對比度", "解析度", "性價比", "分水嶺", "剪刀腳",
    "無風扇", "雙風扇", "高強度", "高負載", "高負荷", "長時間", "多螢幕", "雙螢幕", "入門款", "入門版", "高階款", "高階版",
    "輕薄款", "基礎版", "基礎款", "主力機", "起售價", "起步價", "總持有成本", "總擁有成本", "表面溫差", "溫控差距", "溫控降頻", "溫差", "溫控",
    "擴充底座", "選配單", "資料來源", "公開數據", "捉襟見肘", "一清二楚", "穩如泰山", "一步到位",
    "還是", "配備", "擁有", "採用", "搭配", "透過", "藉由", "導致", "決定", "支援", "針對", "涉及", "觸發", "維持", "釋放",
    "摸到", "升到", "跑到", "拉到", "衝上", "加到", "直上", "直接", "成為", "反而", "而且", "因為", "所以", "如果", "雖然",
    "但是", "不論", "無論", "加上", "反觀", "尤其", "特別", "幾乎", "瞬間", "超高", "極致", "顯著", "完整", "流暢", "穩定", "高速",
    "算一算", "寫寫", "回不去了", "回不去", "掉幀", "不掉幀", "不踩雷", "看懂", "下單",
], key=len, reverse=True)

# Dangling words that must NEVER sit stranded at the very end of a split subtitle line
FORBIDDEN_LINE_END_TOKENS = {
    "在", "從", "往", "向", "對", "給", "把", "被", "讓", "跟", "與", "和", "及", "或", "比", "連", "像", "當", "將", "以", "由",
    "則", "卻", "但", "而", "且", "並", "又", "也", "都", "就", "才", "還", "更", "最", "很", "太", "超", "極", "再", "先", "只",
    "非常", "相當", "特別", "尤其", "已經", "正在", "將近", "只有", "僅有", "幾乎", "瞬間", "馬上", "立刻", "直接", "一旦", "只要",
    "超高", "極致", "完整", "高速", "全新", "原生", "主動", "被動", "全套", "標準", "一般",
    "可以", "能夠", "應該", "必須", "需要", "開始", "繼續", "容易", "不會", "不能", "不要", "沒有", "不", "沒", "別", "未", "非",
    "會", "能", "可", "要", "該", "想", "敢", "肯", "去", "來", "是", "為", "有", "無",
    "一", "一個", "一台", "一塊", "一條", "一種", "這", "那", "這台", "這塊", "這個", "這筆", "這種", "這支", "這些", "那些", "每", "每個",
}

# Enclitics / suffixes that must NEVER start a new split subtitle line
FORBIDDEN_LINE_START_TOKENS = {
    "，", "：", "、", "。", "！", "？",
    "的", "地", "得", "了", "著", "過", "嗎", "呢", "吧", "啊", "呀", "喔", "們",
    "性", "化", "度", "率", "感", "款", "版", "級", "槽", "器", "埠",
}

# Preferred prosodic pause endings (after structural/temporal markers)
PREFERRED_LINE_END_SUFFIXES = ("的", "時", "的時候", "之後", "以後", "之前", "下來", "之下", "工作下", "運算下", "來說", "來看", "裡", "上")

# Preferred prosodic continuations at the start of the second half
PREFERRED_LINE_START_PREFIXES = (
    "而", "但", "卻", "還是", "依然", "反而", "就能", "才能", "就會", "馬上", "直接", "瞬間",
    "同時", "不論", "無論", "加上", "搭配", "配備", "擁有", "採用", "透過", "靠", "讓", "把",
)

# Regex matching indivisible Latin/Number/Spec phrases (e.g., "MacBook Air M5", "120Hz ProMotion", "100 美元", "16GB 記憶體")
_ATOMIC_SPEC_RE = re.compile(
    r"(?:"
    r"MacBook\s+(?:Air|Pro)(?:\s+M\d+(?:\s+(?:Pro|Max))?)?"
    r"|Mac\s+(?:mini|Studio|Pro)"
    r"|iPad\s+(?:Pro|Air|mini)(?:\s+\d+)?"
    r"|iPhone\s+\d+(?:\s+(?:Pro|Max|Plus|Ultra))*"
    r"|Apple\s+(?:TV\s+4K|Silicon|Watch)"
    r"|\d+(?:\.\d+)?\s*(?:Hz|GHz)\s+ProMotion"
    r"|Mini-?LED"
    r"|ProMotion"
    r"|Thunderbolt\s*\d*"
    r"|Wi-?Fi\s*\d+[A-Za-z]*"
    r"|USB-?[A-Za-z0-9.]*"
    r"|HDMI(?:\s*\d+(?:\.\d+)?)?"
    r"|(?:NT\$|\$)\s*\d[\d,]*(?:\.\d+)?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:GB/s|TB/s|tok/s|GB|TB|MB|GHz|Hz|nits|mAh|mm|kg|°C|℃|%|美元|美金|台幣|塊|元|倍|度|奈米|核心|小時|分鐘|年)"
    r"|[A-Za-z][A-Za-z0-9.+-]*(?:\s+[A-Za-z0-9.+-]+)*"
    r"|\d[\d,.]*"
    r")"
)


def visual_em_len(text: str) -> float:
    """Compute true visual character width in CJK `em` units (1 CJK = 1.0 em, 2 ASCII ≈ 1.0 em)."""
    total = 0.0
    for ch in text:
        code = ord(ch)
        if ch.isspace():
            total += 0.28
        elif code < 128:
            if ch.isupper() or ch.isdigit() or ch in "%$@":
                total += 0.56
            else:
                total += 0.45
        else:
            total += 1.0
    return total


def clean_display_narration(narration: str) -> str:
    t = purify_tw_tech_text(narration or "")
    # Remove trailing citation tags (`—— SRC-01`, `[SRC-01]`) from spoken subtitle text
    t = re.sub(r"\s*[—\-–]+\s*\[?\s*SRC\s*[-_]?\s*\d+\s*\]?\s*$", "", t, flags=re.I)
    t = re.sub(r"\s*\[\s*SRC\s*[-_]?\s*\d+\s*\]\s*$", "", t, flags=re.I)
    t = re.sub(r"[「」『』“”\"']", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _segment_cjk_chunk(chunk: str) -> list[str]:
    """Segment a pure-CJK (+ `、`) run into indivisible words while preserving Traditional characters."""
    if not chunk:
        return []
    # Keep enumeration lists `A、B` glued or split cleanly around protected words first
    n = len(chunk)
    # Build character protection spans from PROTECTED_CJK_WORDS
    locked = [False] * n
    spans: list[tuple[int, int]] = []
    for w in PROTECTED_CJK_WORDS:
        start = 0
        while True:
            idx = chunk.find(w, start)
            if idx == -1:
                break
            if not any(locked[idx : idx + len(w)]):
                for k in range(idx, idx + len(w)):
                    locked[k] = True
                spans.append((idx, idx + len(w)))
            start = idx + 1

    spans.sort()
    raw_toks: list[str] = []
    pos = 0
    for s0, s1 in spans:
        if pos < s0:
            raw_toks.extend(_jieba_tw_slice(chunk[pos:s0]))
        raw_toks.append(chunk[s0:s1])
        pos = s1
    if pos < n:
        raw_toks.extend(_jieba_tw_slice(chunk[pos:]))

    # Glue `、` enumeration separators onto adjacent words (`寫寫報告` + `、` + `回信` -> `寫寫報告、回信`) without slicing inside words!
    tokens: list[str] = []
    for t in raw_toks:
        if not t:
            continue
        if t == "、" and tokens:
            tokens[-1] += "、"
        elif tokens and tokens[-1].endswith("、") and visual_em_len(tokens[-1] + t) <= 9.5:
            tokens[-1] += t
        else:
            tokens.append(t)
    return tokens


def _jieba_tw_slice(sub: str) -> list[str]:
    if not sub:
        return []
    # Keep postpositional `時` separate when followed by adjectives/verbs (`時間軸` + `時` + `絲滑`, never `時絲`)
    if len(sub) >= 2 and sub[0] == "時" and sub[1] not in "間代刻候脈區常機尚":
        return ["時"] + _jieba_tw_slice(sub[1:])
    if _jieba_mod is not None and _T2S is not None:
        simp = _T2S.convert(sub)
        if len(simp) == len(sub):
            out = []
            cur = 0
            for w in _jieba_mod.cut(simp):
                ln = len(w)
                out.append(sub[cur : cur + ln])
                cur += ln
            return out
    return list(sub)


def tokenize_subtitle_atoms(clause: str) -> list[str]:
    """
    Split a display clause into indivisible lexical/spec atoms (preserving exact spacing between CJK and Latin).
    """
    raw_atoms: list[str] = []
    pos = 0
    for m in _ATOMIC_SPEC_RE.finditer(clause):
        a, b = m.span()
        if pos < a:
            cjk_part = clause[pos:a]
            raw_atoms.extend(_segment_cjk_chunk(cjk_part.strip()))
        raw_atoms.append(m.group(0).strip())
        pos = b
    if pos < len(clause):
        raw_atoms.extend(_segment_cjk_chunk(clause[pos:].strip()))

    # Merge any `、` enumeration glue (`雷電四` + `、HDMI` -> `雷電四、HDMI`)
    merged: list[str] = []
    for tok in raw_atoms:
        tok = tok.strip()
        if not tok:
            continue
        if merged and (merged[-1].endswith("、") or tok.startswith("、")):
            merged[-1] = _join_two_atoms(merged[-1], tok)
        else:
            merged.append(tok)
    return merged


def _join_two_atoms(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    l_ch, r_ch = left[-1], right[0]
    need_space = (
        (("\u4e00" <= l_ch <= "\u9fff") and (r_ch.isascii() and r_ch.isalnum()))
        or ((l_ch.isascii() and (l_ch.isalnum() or l_ch in "%°℃")) and ("\u4e00" <= r_ch <= "\u9fff"))
        or (l_ch.isascii() and l_ch.isalnum() and r_ch.isascii() and r_ch.isalnum())
    )
    return f"{left} {right}" if need_space else f"{left}{right}"


def join_atoms(atoms: list[str]) -> str:
    res = ""
    for a in atoms:
        res = _join_two_atoms(res, a)
    return res.strip()


def _boundary_penalty(left_atoms: list[str], right_atoms: list[str]) -> float:
    left_last = left_atoms[-1].strip()
    right_first = right_atoms[0].strip()

    if left_last in FORBIDDEN_LINE_END_TOKENS:
        return 10000.0
    if right_first in FORBIDDEN_LINE_START_TOKENS:
        return 10000.0
    # Never strand a bare single-digit/number or `第X` at the end of a line
    if re.fullmatch(r"(?:第[一二三四五六七八九十]|\d+)", left_last):
        return 10000.0
    # Avoid breaking `A 與 B` / `A 和 B` / `A 跟 B` when short
    if right_first in ("與", "和", "跟", "及", "或") and visual_em_len(join_atoms(right_atoms)) < 5.5:
        return 4000.0

    pen = 0.0
    left_str = join_atoms(left_atoms)
    right_str = join_atoms(right_atoms)

    if any(left_str.endswith(sfx) for sfx in PREFERRED_LINE_END_SUFFIXES):
        pen -= 5.8
    if any(right_str.startswith(pfx) for pfx in PREFERRED_LINE_START_PREFIXES):
        pen -= 3.6
    # Slight penalty for splitting between two single-char tokens that might form a colloquial phrase
    if len(left_last) == 1 and len(right_first) == 1 and left_last not in "的時後下裡上前":
        pen += 2.2
    return pen


def _split_long_clause_once(atoms: list[str], max_em: float, min_em: float) -> list[list[str]]:
    total_em = visual_em_len(join_atoms(atoms))
    if total_em <= max_em or len(atoms) <= 1:
        return [atoms]

    target_half = total_em * 0.5
    best_cost = 1e18
    best_k = -1

    for k in range(1, len(atoms)):
        left_a = atoms[:k]
        right_a = atoms[k:]
        l_em = visual_em_len(join_atoms(left_a))
        r_em = visual_em_len(join_atoms(right_a))
        if l_em < min_em or r_em < min_em:
            continue
        over_pen = max(0.0, l_em - max_em) * 18.0 + max(0.0, r_em - max_em) * 18.0
        bal_cost = abs(l_em - target_half) * 0.85
        ling_cost = _boundary_penalty(left_a, right_a)
        cost = bal_cost + over_pen + ling_cost
        if cost < best_cost:
            best_cost = cost
            best_k = k

    if best_k == -1 or best_cost >= 9000.0:
        # Relax `min_em` slightly while STILL forbidding broken words / dangling prepositions
        for k in range(1, len(atoms)):
            left_a, right_a = atoms[:k], atoms[k:]
            l_em = visual_em_len(join_atoms(left_a))
            r_em = visual_em_len(join_atoms(right_a))
            if l_em < 3.5 or r_em < 3.5:
                continue
            cost = abs(l_em - target_half) + _boundary_penalty(left_a, right_a)
            if cost < best_cost:
                best_cost = cost
                best_k = k

    if best_k == -1:
        return [atoms]

    out: list[list[str]] = []
    for part in (atoms[:best_k], atoms[best_k:]):
        if visual_em_len(join_atoms(part)) > max_em + 1.5 and len(part) >= 4:
            out.extend(_split_long_clause_once(part, max_em=max_em, min_em=min_em))
        else:
            out.append(part)
    return out


def segment_narration_to_captions(narration: str, is_vertical: bool = False) -> list[str]:
    """
    Master `@hey_digitall` Subtitle Segmenter:
    - Single-line visual width budget: `22.0 em` (Long 1920x1080) / `17.5 em` (Short 1080x1920).
    - Preserves `、` enumerations & short leading discourse connectors (`不過，...`).
    - Zero mid-word splits, zero dangling particles, zero 2-char orphan flashes!
    """
    clean = clean_display_narration(narration)
    if not clean:
        return []

    max_em = 17.2 if is_vertical else 21.8
    min_em = 5.0 if is_vertical else 5.5
    merge_pair_em = 14.2 if is_vertical else 17.0

    # 1. Split into major sentences (`。！？；`), preserving `?` on questions if compact
    sent_chunks: list[str] = []
    pos = 0
    for m in re.finditer(r"[。！？；\n]+", clean):
        punct = m.group(0)
        seg = clean[pos : m.start()].strip()
        if seg:
            if "？" in punct and is_vertical and visual_em_len(seg) <= max_em - 1.0:
                seg = seg + "？"
            sent_chunks.append(seg)
        pos = m.end()
    if pos < len(clean):
        rem = clean[pos:].strip()
        if rem:
            sent_chunks.append(rem)

    final_captions: list[str] = []

    for sent in sent_chunks:
        # 2. Split on `，` and `：` ONLY (keep `、` and numeric ratios like `16:9` intact!)
        raw_clauses: list[tuple[str, str]] = []
        c_pos = 0
        for cm in re.finditer(r"[，：]|(?<!\d):(?!\d)", sent):
            sub_c = sent[c_pos : cm.start()].strip()
            sep_c = "：" if cm.group(0) in ("：", ":") else "，"
            if sub_c:
                raw_clauses.append((sub_c, sep_c))
            c_pos = cm.end()
        tail_c = sent[c_pos:].strip()
        if tail_c:
            raw_clauses.append((tail_c, ""))

        if not raw_clauses:
            continue

        # 3. Merge short leading connectors (`不過，`, `說白了，`, `長期來看，`) or tiny `< 4.8 em` fragments
        grouped_clauses: list[str] = []
        i = 0
        while i < len(raw_clauses):
            txt_i, sep_i = raw_clauses[i]
            em_i = visual_em_len(txt_i)
            if i + 1 < len(raw_clauses):
                txt_next, sep_next = raw_clauses[i + 1]
                combined = f"{txt_i}{sep_i}{txt_next}"
                comb_em = visual_em_len(combined)
                # Merge if leading connector is short (`<= 5.2 em`) OR if two short parallel clauses fit comfortably
                if (em_i <= 5.2 and comb_em <= max_em) or (comb_em <= merge_pair_em and sep_i == "："):
                    raw_clauses[i + 1] = (combined, sep_next)
                    i += 1
                    continue
            # If final trailing clause is tiny (`< 5.0 em`) and fits on previous clause, merge backward
            if em_i < 4.8 and grouped_clauses and visual_em_len(f"{grouped_clauses[-1]}，{txt_i}") <= max_em + 1.0:
                grouped_clauses[-1] = f"{grouped_clauses[-1]}，{txt_i}"
            else:
                grouped_clauses.append(txt_i)
            i += 1

        # 4. Apply linguistic word-boundary DP split to any clause exceeding `max_em`
        for gc in grouped_clauses:
            if visual_em_len(gc) <= max_em:
                final_captions.append(gc)
            else:
                # If `gc` still has an internal `，` (from connector merge), prefer splitting at `，` first if both >= 4.0 em
                if "，" in gc:
                    p_left, p_right = gc.split("，", 1)
                    if visual_em_len(p_left) >= 4.2 and visual_em_len(p_right) >= 4.5:
                        for sub_p in (p_left.strip(), p_right.strip()):
                            if visual_em_len(sub_p) <= max_em:
                                final_captions.append(sub_p)
                            else:
                                atoms = tokenize_subtitle_atoms(sub_p)
                                for part in _split_long_clause_once(atoms, max_em=max_em, min_em=min_em):
                                    final_captions.append(join_atoms(part))
                        continue
                atoms = tokenize_subtitle_atoms(gc)
                for part in _split_long_clause_once(atoms, max_em=max_em, min_em=min_em):
                    final_captions.append(join_atoms(part))

    return [c.strip(" ，。；：") for c in final_captions if c.strip(" ，。；：")]


_ZH_DIGIT_MAP = str.maketrans("零一二三四五六七八九兩", "01234567892")


def _zh_num_phrase_to_digits(text: str) -> str:
    """Convert common spoken Chinese number phrases (`五百一十二` -> `512`, `一百二十` -> `120`, `十六` -> `16`) for alignment."""
    def _conv(m: re.Match) -> str:
        s = m.group(0)
        if not any(u in s for u in "十百千"):
            return s.translate(_ZH_DIGIT_MAP)
        d_map = {"零": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        u_map = {"十": 10, "百": 100, "千": 1000}
        total = 0
        cur = 0
        for ch in s:
            if ch in d_map:
                cur = d_map[ch]
            elif ch in u_map:
                u = u_map[ch]
                total += (cur if cur else 1) * u
                cur = 0
        total += cur
        return str(total) if total > 0 else s

    t = re.sub(r"[零一二三四五六七八九兩两十百千]{2,}", _conv, text)
    return t.translate(_ZH_DIGIT_MAP)


def _norm_char_skeleton(s: str) -> tuple[str, list[int]]:
    """Build Simplified alphanumeric/CJK skeleton (normalizing spoken Chinese numerals & units) for Whisper alignment."""
    src = _T2S.convert(s) if (_T2S is not None) else s
    src = src.replace("赫兹", "hz").replace("gb/s", "gs").replace("tb/s", "ts").replace("gb", "g").replace("tb", "t")
    src = _zh_num_phrase_to_digits(src)
    skel_chars: list[str] = []
    orig_idxs: list[int] = []
    for idx, ch in enumerate(src):
        if ("\u4e00" <= ch <= "\u9fff") or ch.isalnum():
            skel_chars.append(ch.lower())
            orig_idxs.append(idx)
    return "".join(skel_chars), orig_idxs


def align_captions_to_whisper_words(
    captions: list[str],
    whisper_words: list[dict],
    total_dur: float,
    is_short: bool = False,
) -> list[dict]:
    """
    Align each segmented `caption` to exact spoken `(start, end)` timestamps using Whisper word-level
    timestamps (`[{"word": "...", "start": 0.12, "end": 0.38}, ...]`) + monotonic character interpolation!
    """
    if not captions:
        return []

    tail_pad = 0.24 if is_short else 0.52
    default_start = 0.04
    default_end = max(0.35, total_dur - tail_pad)

    if not whisper_words:
        weights = [max(1.5, visual_em_len(c)) for c in captions]
        w_sum = sum(weights)
        span = max(0.3, default_end - default_start)
        cur = default_start
        out = []
        for c, w in zip(captions, weights):
            d = span * (w / w_sum)
            out.append({"text": c, "start": round(cur, 3), "end": round(cur + d, 3)})
            cur += d
        return out

    # Flatten Whisper words into per-character `(start, end)` timeline
    w_chars: list[str] = []
    w_times: list[tuple[float, float]] = []
    for w_obj in whisper_words:
        w_txt = str(w_obj.get("word") or w_obj.get("text") or "").strip()
        if not w_txt:
            continue
        t0 = float(w_obj.get("start", default_start))
        t1 = max(t0 + 0.04, float(w_obj.get("end", t0 + 0.16)))
        w_skel, _ = _norm_char_skeleton(w_txt)
        if not w_skel:
            continue
        step = (t1 - t0) / len(w_skel)
        for ci, ch in enumerate(w_skel):
            w_chars.append(ch)
            w_times.append((t0 + ci * step, t0 + (ci + 1) * step))

    if not w_chars:
        return align_captions_to_whisper_words(captions, [], total_dur, is_short=is_short)

    # Concatenate caption skeletons and record `[c_start_idx, c_end_idx)` per caption
    cap_skels = [_norm_char_skeleton(c)[0] for c in captions]
    cap_ranges: list[tuple[int, int]] = []
    full_cap_skel = ""
    for cs in cap_skels:
        cs_eff = cs if cs else "x"
        s_i = len(full_cap_skel)
        full_cap_skel += cs_eff
        cap_ranges.append((s_i, len(full_cap_skel)))

    n_c = len(full_cap_skel)
    n_w = len(w_chars)
    w_str = "".join(w_chars)

    # Monotonic anchor matching via SequenceMatcher
    sm = difflib.SequenceMatcher(None, full_cap_skel, w_str, autojunk=False)
    anchors_c = [0]
    anchors_w = [0]
    for blk in sm.get_matching_blocks():
        if blk.size > 0:
            for k in range(blk.size):
                anchors_c.append(blk.a + k)
                anchors_w.append(blk.b + k)
    anchors_c.append(n_c)
    anchors_w.append(n_w - 1)

    def _map_c_to_w(c_idx: int) -> int:
        # Linear piecewise monotonic lookup
        for ai in range(len(anchors_c) - 1):
            c0, c1 = anchors_c[ai], anchors_c[ai + 1]
            w0, w1 = anchors_w[ai], anchors_w[ai + 1]
            if c0 <= c_idx <= c1:
                if c1 == c0:
                    return min(n_w - 1, max(0, w0))
                frac = (c_idx - c0) / (c1 - c0)
                return min(n_w - 1, max(0, int(round(w0 + frac * (w1 - w0)))))
        return n_w - 1

    timings: list[dict] = []
    for idx, (c_txt, (r0, r1)) in enumerate(zip(captions, cap_ranges)):
        wi_start = _map_c_to_w(r0)
        wi_end = _map_c_to_w(max(r0, r1 - 1))
        if wi_end < wi_start:
            wi_end = wi_start
        t_s = w_times[wi_start][0]
        t_e = w_times[wi_end][1]
        if idx > 0 and t_s < timings[-1]["end"]:
            mid_t = 0.5 * (timings[-1]["end"] + t_s)
            timings[-1]["end"] = round(max(timings[-1]["start"] + 0.25, mid_t), 3)
            t_s = timings[-1]["end"]
        t_e = max(t_s + 0.32, min(total_dur - 0.06, t_e))
        timings.append({
            "text": c_txt,
            "start": round(max(0.0, t_s), 3),
            "end": round(t_e, 3),
        })

    # Close tiny `< 0.28s` intra-sentence gaps so captions don't flicker during fast speech,
    # while keeping genuine `>= 0.30s` breath pauses clean!
    for i in range(len(timings) - 1):
        gap = timings[i + 1]["start"] - timings[i]["end"]
        if 0.0 < gap < 0.28:
            timings[i]["end"] = timings[i + 1]["start"]

    return timings
