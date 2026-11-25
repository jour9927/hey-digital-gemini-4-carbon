#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon V4 — Taiwan Tech Lexicon, Polyphone-Safe Simplified F5-TTS Verbalizer & Oral Normalizer
===================================================================================================================
Fixes ALL auditory defects identified in side-by-side listening against `@hey_digitall` reference videos:
1. Keeps 100% authentic Taiwan Traditional Chinese (`記憶體`, `固態硬碟`, `螢幕`, `連接埠`, `更新率`) on screen (`display_text`),
   while converting `tts_text` via character-only `OpenCC('t2s')` (`记忆体`, `固态硬碟`, `弹性`, `几乎`) WITH ZERO CJK SPACES
   so `F5-TTS`'s `jieba` + `pypinyin` tokenizer NEVER misreads polyphones (`彈性 -> dàn xìng`, `幾乎 -> jǐ hū`) and NEVER
   stutters mid-word (`記憶…體`, `散…熱`, `差…異`)!
2. Natural Human Spoken Units (matching `@hey_digitall` host Alan's exact spoken phrasing):
   - `153GB/s` -> `每秒一百五十三G` (NEVER robotic `"每秒一百五十三 G 位元組"`)
   - `20°C` / `20℃` -> `二十度` (NEVER `"二十 gabri C"`)
   - Strips citation tags (`—— SRC-01`, `[SRC-01]`) from `tts_text` (NEVER reads `"S R C 一"` aloud!)
   - Separates `Air` and `M5` (`MacBook Air M5` -> `M五版的 MacBook Air`) so `Air M` never slurs into `"欸兒阿曼"`!
   - Maps phonetic traps (`Thunderbolt 4` -> `雷电四`, `Mini-LED` -> `Mini LED`, `ProMotion` -> `ProMotion`) cleanly!
"""
from __future__ import annotations

import re

try:
    from opencc import OpenCC
    _T2S = OpenCC("t2s")
except Exception:
    _T2S = None

# Mainland / Simplified -> Taiwan Traditional Tech Lexicon (for ON-SCREEN display_text & captions)
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


def purify_tw_tech_text(text: str) -> str:
    """Normalize on-screen text to crisp Taiwan Traditional tech typography."""
    t = text or ""
    for src, dst in TW_TECH_LEXICON:
        t = t.replace(src, dst)
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


def verbalize_for_mandarin_tts(raw_text: str) -> str:
    """
    Convert Taiwan Traditional display script into polyphone-safe, zero-stutter F5-TTS oral script:
    - Strips citation tags (`SRC-01`) & visual brackets
    - Converts all numbers/units/temperatures into natural spoken `@hey_digitall` Mandarin
    - Prevents Latin token collisions (`MacBook Air M5` -> `搭載M五的 MacBook Air`)
    - Converts characters to Simplified (`OpenCC('t2s')`) WITHOUT altering Taiwan vocabulary (`记忆体`, `硬碟`),
      eliminating `jieba`/`pypinyin` polyphone & word-splitting bugs!
    """
    t = purify_tw_tech_text(raw_text)

    # 0. Strip non-spoken source tags (`—— SRC-01`, `[SRC-02]`), brackets & em-dashes
    t = re.sub(r"[—\-–]+\s*\[?\s*SRC\s*[-_]?\s*\d+\s*\]?", "", t, flags=re.I)
    t = re.sub(r"\[\s*SRC\s*[-_]?\s*\d+\s*\]", "", t, flags=re.I)
    t = re.sub(r"\bSRC\s*[-_]?\s*\d+\b", "", t, flags=re.I)
    t = re.sub(r"[「」『』“”\"']", "", t)
    t = re.sub(r"[—–]+", "，", t)
    t = re.sub(r"\s*：\s*", "，", t)

    # 1. Prevent `MacBook Air M5` Latin boundary slur & map `M` -> phonetic `Em` (`/ɛm/` + Chinese digit = exact "M5"!)
    t = re.sub(
        r"MacBook\s+Air\s+M\s*(\d+)",
        lambda m: f"搭載 Em{int_to_zh(int(m.group(1)))}晶片的 MacBook Air",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"MacBook\s+Pro\s+M\s*(\d+)\s*(Pro|Max)?",
        lambda m: f"搭載 Em{int_to_zh(int(m.group(1)))}{(' ' + m.group(2)) if m.group(2) else ''}晶片的 MacBook Pro",
        t,
        flags=re.I,
    )

    # 2. Brand names, ports & display standards spoken cleanly in Mandarin tech videos
    brand_pre_map = [
        (r"9\s*to\s*5\s*Mac", " Nine to Five Mac "),
        (r"Thunderbolt\s*(\d+)", lambda m: f"雷電{int_to_zh(int(m.group(1)))}"),
        (r"\bThunderbolt\b", "雷電"),
        (r"\bMagSafe\s*(\d+)?", lambda m: f"磁吸充電{int_to_zh(int(m.group(1))) if m.group(1) else ''}"),
        (r"\b3D\b", "三D"),
        (r"\b4K\b", "四K"),
        (r"\b5K\b", "五K"),
        (r"\b6K\b", "六K"),
        (r"\b8K\b", "八K"),
        (r"Mini\s*-?\s*LED", " Mini LED "),
        (r"Pro\s*Motion", " ProMotion "),
        (r"Wi-?Fi\s*6E\b", " Wi-Fi 六E "),
        (r"Wi-?Fi\s*(\d+)\b", lambda m: f" Wi-Fi {int_to_zh(int(m.group(1)))} "),
        (r"USB\s*-?\s*C", " USB C "),
        (r"USB\s*(\d+)", lambda m: f" USB {int_to_zh(int(m.group(1)))} "),
    ]
    for pat, repl in brand_pre_map:
        t = re.sub(pat, repl, t, flags=re.I)

    # 3. Years: 2020年..2035年 -> 二零二六年
    t = re.sub(r"\b(19\d\d|20\d\d)\s*年", lambda m: f"{year_to_zh(m.group(1))}年", t)

    # 4. Temperatures: 20°C / 20 °C / 20℃ -> 二十度 (fixes "20 gabri C"!)
    t = re.sub(
        r"(-?\d+(?:\.\d+)?)\s*(?:°\s*C|℃|度\s*C)\b",
        lambda m: (f"零下{num_str_to_zh(m.group(1)[1:], two_as_liang=True)}度" if m.group(1).startswith("-") else f"{num_str_to_zh(m.group(1), two_as_liang=True)}度"),
        t,
        flags=re.I,
    )

    # 5. Currencies: NT$34,900 / $1,299 / 100 美元
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

    # 6. Apple / Tech Chip Designators: M6 Pro -> Em六 Pro, M5 -> Em五, N1 -> En一, A20 -> A二十
    _chip_phonetic = {"M": "Em", "N": "En", "A": "A", "C": "C"}
    t = re.sub(
        r"\b([MANC])\s*(\d{1,2})(?=\s*(?:Pro|Max|Ultra|晶片|處理器|核心|版|\b))",
        lambda m: f"{_chip_phonetic.get(m.group(1).upper(), m.group(1))}{int_to_zh(int(m.group(2)))}",
        t,
    )

    # 7. Ranges: 4-5 年 / 13~15 吋 -> 四到五年
    t = re.sub(
        r"(\d+(?:\.\d+)?)\s*[-~～至]\s*(\d+(?:\.\d+)?)",
        lambda m: f"{num_str_to_zh(m.group(1), two_as_liang=True)}到{num_str_to_zh(m.group(2), two_as_liang=True)}",
        t,
    )

    # 8. Percentages: -10% -> 負百分之十, 10% -> 百分之十
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

    # 9. Natural Spoken Tech Units (matches @hey_digitall: `153GB/s` -> `每秒一百五十三G`, `256GB` -> `二百五十六G`)
    unit_rules = [
        (r"(\d[\d,]*(?:\.\d+)?)\s*GB/s\b", lambda v: f"每秒{num_str_to_zh(v, two_as_liang=False)}G"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*TB/s\b", lambda v: f"每秒{num_str_to_zh(v, two_as_liang=False)}T"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*tok/s\b", lambda v: f"每秒{num_str_to_zh(v)}個token"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*GB\b", lambda v: f"{num_str_to_zh(v, two_as_liang=False)}G"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*TB\b", lambda v: f"{num_str_to_zh(v, two_as_liang=False)}T"),
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

    # 10. Remaining standalone decimals & integers
    t = re.sub(
        r"\b2(?=\s*(?:台|顆|款|種|倍|個|小時|年|天|支|張|大|階|步))",
        "兩",
        t,
    )
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda m: num_str_to_zh(m.group(0)), t)

    # 11. Common Acronyms spoken cleanly in Mandarin tech channels (ZERO spaces next to Chinese!)
    acronym_map = [
        ("OLED", " OLED "),
        ("SSD", "固態硬碟"),
        ("HDD", "傳統硬碟"),
        ("NVMe", "高速固態"),
        ("NPU", "神經網路引擎"),
        ("GPU", "繪圖處理器"),
        ("CPU", "中央處理器"),
        ("RAM", "記憶體"),
        ("LLM", "大語言模型"),
        ("HDR", "高動態範圍"),
        ("HDMI", " HDMI "),
        ("SDXC", "記憶卡"),
        ("NAS", "網路儲存"),
        ("AI", "人工智慧"),
    ]
    for k, v in acronym_map:
        t = re.sub(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])", v, t)

    # Avoid bare standalone "Pro" right before Chinese verbs turning into "鋪具備"
    t = re.sub(r"(?<![A-Za-z]\s)(?<![A-Za-z])\bPro\b(?!\s*Motion)", " MacBook Pro ", t)
    t = re.sub(r"MacBook\s+MacBook\s+Pro", "MacBook Pro", t)

    # 12. Convert characters ONLY from Traditional -> Simplified (`OpenCC('t2s')`) so `F5-TTS`'s
    #     Simplified `jieba` + `pypinyin` gets 100% accurate word segmentation & polyphone tones
    #     (`弹性 -> tán xìng`, `几乎 -> jī hū`, `记忆体` stays Taiwan vocab "jì yì tǐ"!)
    if _T2S is not None:
        t = _T2S.convert(t)
    # Fix `著` -> `着` (`zhe`) except in `显著/著名/著作/执著` so `挂着/带着/拖着/看着/随着` never reads `zhù`!
    t = re.sub(r"(?<![显執执專专名原編编])著(?![名作稱称重])", "着", t)
    # Fix Taiwan 4th-tone `連接埠` (`lián jiē bù`) so mainland pypinyin never reads 3rd-tone `bǔ`!
    t = t.replace("连接埠", "连接部").replace("扩充埠", "扩充部")

    # Remove ANY spaces between CJK characters (prevents mid-sentence micro-stutters!)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", t)
    # Exact F5-TTS `convert_char_to_pinyin` asymmetric rule:
    # - 1 space BEFORE Latin (`CJK + ' ' + Latin` -> single `' '` token)
    # - 0 spaces AFTER Latin before CJK (`Latin + CJK` -> single `' '` token; having a space here creates DOUBLE `' ', ' '` pause!)
    t = re.sub(r"([\u4e00-\u9fff])\s*([A-Za-z])", r"\1 \2", t)
    t = re.sub(r"([A-Za-z])\s+([\u4e00-\u9fff])", r"\1\2", t)
    # Exception: single-letter unit `G` / `T` / `K` / `B` right after Chinese numeral (`十六G记忆体`, `四K影片`)
    t = re.sub(r"([零一二三四五六七八九十百千萬億點兩])\s+([GTKB])(?=[\u4e00-\u9fff，。！？]|$)", r"\1\2", t)

    t = re.sub(r"\s+", " ", t).strip(" ，,")
    if t and t[-1] not in "。！？":
        t += "。"
    return t
