#!/usr/bin/env python3
"""
Hey Digital V6 — 1:1 Visual-Audio Aligned Taiwan Tech Lexicon & F5/Neural TTS Oral Normalizer
==============================================================================================
Fixes ALL prosodic & phonetic defects audited against `@hey_digitall` reference videos (`D9IiQF_QIYg`, `1TO7dzrTuQg`, `QrBoNy0wse8`):
1. 1:1 Subtitle-to-Speech Word Parity (ZERO secret word additions or substitutions):
   - `MacBook Air M5` stays `MacBook Air M五` (NEVER rewritten to `"搭載 Em五晶片的 MacBook Air"`, and NEVER `Em五`!).
   - `SSD` -> `固態硬碟` and `RAM` -> `記憶體` happen in `purify_tw_tech_text` so on-screen captions and spoken audio match 1:1!
   - Standalone `Air` and `Pro` stay `Air` and `Pro` (NEVER rewritten to `MacBook Pro`).
   - Spoken acronyms (`AI`, `NAS`, `OLED`, `HDMI`, `USB-C`, `Mini-LED`, `ProMotion`, `Wi-Fi 7`) stay aligned between screen and voice.
2. Phonetic Clarity & Natural `@hey_digitall` Spoken Units:
   - `M5` / `M6` -> `M五` / `M六` (matches exact single-letter `M` token in clean reference prompt!).
   - `1TB` -> `一 TB` (prevents `一T` slurring into `"ET"`), `16GB` -> `十六G`, `153GB/s` -> `每秒一百五十三G`.
   - `120Hz` -> `一百二十赫茲`, `20°C` -> `二十度`, `2026年` -> `二零二六年`.
   - Keeps `連接埠` intact (`埠` is already 4th-tone `bu4`; never corrupts to `连接部`).
"""
from __future__ import annotations

import re

try:
    from opencc import OpenCC
    _T2S = OpenCC("t2s")
except Exception:
    _T2S = None

# Mainland / Simplified -> Taiwan Traditional Tech Lexicon (applied to BOTH display_text & tts_text)
TW_TECH_LEXICON = [
    ("内存带宽", "記憶體頻寬"), ("统一内存", "統一記憶體"), ("内存", "記憶體"), ("运存", "統一記憶體"), ("显存", "顯示記憶體"),
    ("带宽", "頻寬"), ("固态硬盘", "固態硬碟"), ("硬盘", "硬碟"),
    ("显示屏", "螢幕"), ("屏幕", "螢幕"), ("刷新率", "更新率"), ("高刷", "高更新率"),
    ("笔记本电脑", "筆記型電腦"), ("笔记本", "筆電"),
    ("芯片", "晶片"), ("处理器", "處理器"),
    ("视频", "影片"), ("软件", "軟體"), ("硬件", "硬體"), ("网络", "網路"),
    ("算法", "演算法"), ("人工智能", "人工智慧"), ("大模型", "大語言模型"),
    ("接口", "連接埠"), ("充电头", "充電器"),
    ("性价比", "性價比"), ("两百刀", "兩百美元"), ("三百刀", "三百美元"), ("美刀", "美元"),
    ("分辨率", "解析度"), ("像素", "畫素"), ("鼠标", "滑鼠"), ("菜单", "選單"),
    ("默认", "預設"), ("设置", "設定"), ("打开", "開啟"), ("关闭", "關閉"),
    ("支持", "支援"), ("通过", "透過"), ("优化", "最佳化"), ("质量", "品質"),
    ("信息", "資訊"), ("数据", "數據"), ("虽然", "雖然"), ("但是", "但"),
    ("实际", "實際"), ("表现", "表現"), ("体验", "體驗"), ("价格", "價格"),
    ("选购", "選購"), ("对比", "對比"), ("评测", "評測"), ("测试", "實測"),
    ("续航", "續航"), ("发热", "發熱"), ("散热", "散熱"), ("轻薄", "輕薄"),
    ("生态", "生態"), ("扩展", "擴充"), ("文件", "檔案"),
    ("点击", "點擊"), ("链接", "連結"), ("订阅", "訂閱"), ("点赞", "按讚"),
    ("这期", "這支"), ("本期视频", "這支影片"), ("更强", "更強"), ("够用", "夠用"),
]

_DIGITS = "零一二三四五六七八九"
_ZH_K_MAP = {"二": "2", "四": "4", "五": "5", "六": "6", "八": "8"}


def purify_tw_tech_text(text: str) -> str:
    """Normalize on-screen text to crisp Taiwan Traditional tech typography (matching spoken lexicon 1:1)."""
    t = text or ""
    for src, dst in TW_TECH_LEXICON:
        t = t.replace(src, dst)
    # Normalize `四 K` / `4 K` -> `4K` on screen
    t = re.sub(r"([二四五六八])\s*K\b", lambda m: f"{_ZH_K_MAP[m.group(1)]}K", t, flags=re.I)
    t = re.sub(r"\b([24568])\s+K\b", r"\1K", t, flags=re.I)
    # Align SSD / RAM on screen with `@hey_digitall`'s natural spoken Mandarin (`512GB 固態硬碟`, `16GB 記憶體`)
    t = re.sub(r"(?<![A-Za-z])SSD(?![A-Za-z])", "固態硬碟", t)
    t = re.sub(r"(?<![A-Za-z])RAM(?![A-Za-z])", "記憶體", t)
    t = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9$])", r"\1 \2", t)
    t = re.sub(r"([A-Za-z0-9%°℃])([\u4e00-\u9fff])", r"\1 \2", t)
    return re.sub(r"\s+", " ", t).strip()


def int_to_zh(n: int, two_as_liang: bool = False) -> str:
    if n == 0:
        return "零"
    if n < 0:
        return "負" + int_to_zh(-n, two_as_liang=two_as_liang)
    if n == 2 and two_as_liang:
        return "兩"

    def _section_4(val: int) -> str:
        units = ["", "十", "百", "千"]
        s = str(val).zfill(4)
        res = []
        zero_flag = False
        for i, ch in enumerate(s):
            d = int(ch)
            pos = 3 - i
            if d == 0:
                zero_flag = True
            else:
                if zero_flag and res:
                    res.append("零")
                if d == 2 and pos >= 2:
                    res.append(("兩" if two_as_liang else "二") + units[pos])
                elif d == 1 and pos == 1 and not res and val < 20:
                    res.append("十")
                else:
                    res.append(_DIGITS[d] + units[pos])
                zero_flag = False
        return "".join(res)

    if n < 10000:
        return _section_4(n)
    if n < 100_000_000:
        hi, lo = divmod(n, 10000)
        hi_s = "兩" if hi == 2 else _section_4(hi)
        if lo == 0:
            return f"{hi_s}萬"
        if lo < 1000:
            return f"{hi_s}萬零{_section_4(lo)}"
        return f"{hi_s}萬{_section_4(lo)}"
    hi, rem = divmod(n, 100_000_000)
    return f"{int_to_zh(hi)}億{int_to_zh(rem) if rem else ''}"


def num_str_to_zh(s: str, two_as_liang: bool = False) -> str:
    s = s.replace(",", "").strip()
    if not s:
        return ""
    if "." in s:
        ip, fp = s.split(".", 1)
        ip_zh = int_to_zh(int(ip or "0"), two_as_liang=False)
        fp_zh = "".join(_DIGITS[int(c)] for c in fp if c.isdigit())
        return f"{ip_zh}點{fp_zh}" if fp_zh else ip_zh
    return int_to_zh(int(s), two_as_liang=two_as_liang)


def year_to_zh(y: str) -> str:
    return "".join(_DIGITS[int(c)] for c in y if c.isdigit())


def verbalize_for_mandarin_tts(raw_text: str, letter_space_acronyms: bool = True) -> str:
    """
    Convert Taiwan Traditional display script into 1:1 word-order aligned, polyphone-safe oral script:
    - Preserves exact word order (`MacBook Air M5` -> `MacBook Air M五`, zero added/swapped words!).
    - Converts numbers/units/temperatures into natural `@hey_digitall` spoken Mandarin.
    - Converts Traditional -> Simplified characters ONLY (`OpenCC('t2s')`) while keeping Taiwan vocabulary
      (`记忆体`, `固态硬碟`, `连接埠`, `萤幕`) for accurate `jieba`/`pypinyin` polyphone disambiguation.
    """
    t = purify_tw_tech_text(raw_text)

    # 0. Strip non-spoken source tags (`—— SRC-01`, `[SRC-02]`), brackets & em-dashes
    t = re.sub(r"[—\-–]+\s*\[?\s*SRC\s*[-_]?\s*\d+\s*\]?", "", t, flags=re.I)
    t = re.sub(r"\[\s*SRC\s*[-_]?\s*\d+\s*\]", "", t, flags=re.I)
    t = re.sub(r"\bSRC\s*[-_]?\s*\d+\b", "", t, flags=re.I)
    t = re.sub(r"[「」『』“”\"']", "", t)
    t = re.sub(r"[—–]+", "，", t)
    t = re.sub(r"\s*：\s*", "，", t)

    # 1. Brand names, ports & display standards spoken cleanly in `@hey_digitall` reference videos
    hdmi_tok = " H D M I " if letter_space_acronyms else " HDMI "
    usb_c_tok = " U S B C " if letter_space_acronyms else " USB-C "
    mini_led_tok = " Mini L E D " if letter_space_acronyms else " Mini-LED "
    ai_tok = " A I " if letter_space_acronyms else " AI "

    brand_pre_map = [
        (r"9\s*to\s*5\s*Mac", " Nine to Five Mac "),
        (r"Thunderbolt\s*(\d+)", lambda m: f" Thunderbolt {int_to_zh(int(m.group(1)))} "),
        (r"\bMagSafe\s*(\d+)?", lambda m: f" MagSafe {int_to_zh(int(m.group(1))) if m.group(1) else ''} "),
        (r"\b3D\b", "三D"),
        (r"\b4K\b", "四K"),
        (r"\b5K\b", "五K"),
        (r"\b6K\b", "六K"),
        (r"\b8K\b", "八K"),
        (r"Mini\s*-?\s*LED", mini_led_tok),
        (r"Pro\s*Motion", " ProMotion "),
        (r"Wi-?Fi\s*6E\b", " Wi-Fi 六E "),
        (r"Wi-?Fi\s*(\d+)\b", lambda m: f" Wi-Fi {int_to_zh(int(m.group(1)))} "),
        (r"USB\s*-?\s*C\b", usb_c_tok),
        (r"USB\s*(\d+)\b", lambda m: f" U S B {int_to_zh(int(m.group(1)))} " if letter_space_acronyms else f" USB {int_to_zh(int(m.group(1)))} "),
        (r"\bHDMI\b", hdmi_tok),
        (r"\bAI\b", ai_tok),
    ]
    for pat, repl in brand_pre_map:
        t = re.sub(pat, repl, t, flags=re.I)

    # 2. Years: 2020年..2035年 -> 二零二六年
    t = re.sub(r"\b(19\d\d|20\d\d)\s*年", lambda m: f"{year_to_zh(m.group(1))}年", t)

    # 3. Temperatures: 20°C / 20 °C / 20℃ -> 二十度
    t = re.sub(
        r"(-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃|度\s*C)\b",
        lambda m: (f"零下{num_str_to_zh(m.group(1)[1:], two_as_liang=True)}度" if m.group(1).startswith("-") else f"{num_str_to_zh(m.group(1), two_as_liang=True)}度"),
        t,
        flags=re.I,
    )

    # 4. Currencies: NT$34,900 / $1,299 / 100 美元
    t = re.sub(
        r"NT\$\s*(\d[\d,]*(?:\.\d+)?)",
        lambda m: f"台幣{num_str_to_zh(m.group(1), two_as_liang=True)}元",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\$\s*(\d[\d,]*(?:\.\d+)?)",
        lambda m: f"{num_str_to_zh(m.group(1), two_as_liang=True)}美元",
        t,
    )
    t = re.sub(
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:USD|美金|美元)\b",
        lambda m: f"{num_str_to_zh(m.group(1), two_as_liang=True)}美元",
        t,
        flags=re.I,
    )

    # 5. Apple / Tech Chip Designators: `M5` -> `M五`, `M6 Pro` -> `M六 Pro`, `N1` -> `N一`, `A18 Pro` -> `A十八 Pro`
    t = re.sub(
        r"\b([MANC])\s*(\d{1,2})(?=\s*(?:Pro|Max|Ultra|晶片|處理器|核心|版|世代|\b))",
        lambda m: f"{m.group(1).upper()}{int_to_zh(int(m.group(2)))}",
        t,
    )

    # 6. Ranges: 4-5 年 / 13~15 吋 -> 四到五年
    t = re.sub(
        r"(\d+(?:\.\d+)?)\s*[-~～至]\s*(\d+(?:\.\d+)?)",
        lambda m: f"{num_str_to_zh(m.group(1), two_as_liang=True)}到{num_str_to_zh(m.group(2), two_as_liang=True)}",
        t,
    )

    # 7. Percentages: -10% -> 負百分之十, 10% -> 百分之十
    t = re.sub(
        r"(?<=[少減降跌])\s*-\s*(\d+(?:\.\d+)?)\s*%",
        lambda m: f"百分之{num_str_to_zh(m.group(1))}",
        t,
    )
    t = re.sub(
        r"-\s*(\d+(?:\.\d+)?)\s*%",
        lambda m: f"負百分之{num_str_to_zh(m.group(1))}",
        t,
    )
    t = re.sub(
        r"\+?\s*(\d+(?:\.\d+)?)\s*%",
        lambda m: f"百分之{num_str_to_zh(m.group(1))}",
        t,
    )

    # 8. Natural Spoken Tech Units (`153GB/s` -> `每秒一百五十三G`, `1TB` -> `一 TB` so `一T` never slurs into `ET`!)
    tb_unit = " T B " if letter_space_acronyms else " TB "
    unit_rules = [
        (r"(\d[\d,]*(?:\.\d+)?)\s*GB/s\b", lambda v: f"每秒{num_str_to_zh(v, two_as_liang=False)}G"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*TB/s\b", lambda v: f"每秒{num_str_to_zh(v, two_as_liang=True)}{tb_unit}"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*tok/s\b", lambda v: f"每秒{num_str_to_zh(v)}個 token "),
        (r"(\d[\d,]*(?:\.\d+)?)\s*GB\b", lambda v: f"{num_str_to_zh(v, two_as_liang=False)}G"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*TB\b", lambda v: f"{num_str_to_zh(v, two_as_liang=True)}{tb_unit}"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*GHz\b", lambda v: f"{num_str_to_zh(v)}G赫茲"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*Hz\b", lambda v: f"{num_str_to_zh(v)}赫茲"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*(?:nits?|尼特)\b", lambda v: f"{num_str_to_zh(v, two_as_liang=True)}尼特"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*mAh\b", lambda v: f"{num_str_to_zh(v, two_as_liang=True)}毫安時"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*mm\b", lambda v: f"{num_str_to_zh(v)}公釐"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*kg\b", lambda v: f"{num_str_to_zh(v)}公斤"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*g\b", lambda v: f"{num_str_to_zh(v)}公克"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*W\b", lambda v: f"{num_str_to_zh(v)}瓦"),
        (r"(\d+(?:\.\d+)?)\s*B(?=\s*(?:模型|參數|\b))", lambda v: f"{num_str_to_zh(v)}B"),
    ]
    for pat, fn in unit_rules:
        t = re.sub(pat, lambda m, f=fn: f(m.group(1)), t, flags=re.I)

    # 9. Remaining standalone decimals & integers
    t = re.sub(
        r"\b2(?=\s*(?:台|顆|款|種|倍|個|小時|年|天|支|張|大|階|步|奈米))",
        "兩",
        t,
    )
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda m: num_str_to_zh(m.group(0)), t)

    # 10. Convert characters ONLY from Traditional -> Simplified (`OpenCC('t2s')`)
    if _T2S is not None:
        t = _T2S.convert(t)
    # Fix `著` -> `着` (`zhe`) except in `显著/著名/著作/执著`
    t = re.sub(r"(?<![显執执專专名原編编])著(?![名作稱称重])", "着", t)

    # Remove spaces between pure CJK characters (prevents mid-word stutters)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", t)
    # Keep clean single spaces around English words so `Air采用` never coarticulates into `A2采用` and `Pro配备` never becomes `Pro Pay被`!
    t = re.sub(r"([\u4e00-\u9fff])\s*([A-Za-z])", r"\1 \2", t)
    t = re.sub(r"([A-Za-z])\s*([\u4e00-\u9fff])", r"\1 \2", t)
    # Keep single-letter unit `G` / `K` / `B` tight after Chinese numeral (`十六G记忆体`, `四K影片`) and `M五` tight (`M五晶片`)
    t = re.sub(r"([零一二三四五六七八九十百千万亿点两])\s+([GKB])\s*(?=[\u4e00-\u9fff，、。！？]|$)", r"\1\2", t)
    t = re.sub(r"\b([MANC])\s+([零一二三四五六七八九十]+)", r"\1\2", t)

    t = re.sub(r"\s+", " ", t).strip(" ，,")
    if t and t[-1] not in "。！？":
        t += "。"
    return t
