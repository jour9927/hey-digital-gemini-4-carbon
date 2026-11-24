#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon V2 — Taiwan Tech Lexicon Purifier & 100% Mandarin TTS Verbalizer
=============================================================================================
Eliminates ALL English number-reading glitches in PyTorch F5-TTS (`"one thousand two ninety nine"`,
`"Man-six"`, `"one twenty Hz"`) by converting every currency, chip model, unit, decimal, range,
year, and integer into natural Taiwanese Mandarin spoken form (`tts_text`), while keeping clean
Apple-Keynote typography in `display_text`!
"""
from __future__ import annotations

import re

# Mainland / Simplified -> Taiwan Traditional Tech Lexicon
TW_TECH_LEXICON = [
    ("内存", "記憶體"), ("运存", "統一記憶體"), ("显存", "顯示記憶體"),
    ("带宽", "頻寬"), ("内存带宽", "記憶體頻寬"),
    ("硬盘", "硬碟"), ("固态硬盘", "固態硬碟"),
    ("屏幕", "螢幕"), ("显示屏", "螢幕"), ("刷新率", "更新率"), ("高刷", "高更新率"),
    ("笔记本电脑", "筆記型電腦"), ("笔记本", "筆電"),
    ("芯片", "晶片"), ("处理器", "處理器"),
    ("视频", "影片"), ("软件", "軟體"), ("硬件", "硬體"), ("网络", "網路"),
    ("算法", "演算法"), ("人工智能", "人工智慧"), ("大模型", "大語言模型"),
    ("接口", "連接埠"), ("雷电", "Thunderbolt "), ("充电头", "充電器"),
    ("性价比", "性價比"), ("两百刀", "兩百美元"), ("三百刀", "三百美元"), ("美刀", "美元"),
    ("分辨率", "解析度"), ("像素", "畫素"), ("鼠标", "滑鼠"), ("菜单", "選單"),
    ("默认", "預設"), ("设置", "設定"), ("打开", "開啟"), ("关闭", "關閉"),
    ("支持", "支援"), ("通过", "透過"), ("优化", "最佳化"), ("质量", "品質"),
    ("信息", "資訊"), ("数据", "數據"), ("虽然", "雖然"), ("但是", "但"),
    ("实际", "實際"), ("表现", "表現"), ("体验", "體驗"), ("价格", "價格"),
    ("选购", "選購"), ("对比", "對比"), ("评测", "評測"), ("测试", "實測"),
    ("续航", "續航"), ("发热", "發熱"), ("散热", "散熱"), ("轻薄", "輕薄"),
    ("生态", "生態"), ("扩展", "擴充"), ("文件", "檔案"), ("夹", "夾"),
    ("点击", "點擊"), ("链接", "連結"), ("订阅", "訂閱"), ("点赞", "按讚"),
    ("这期", "這支"), ("本期视频", "這支影片"), ("更强", "更強"), ("够用", "夠用"),
]

_DIGITS = "零一二三四五六七八九"


def purify_tw_tech_text(text: str) -> str:
    t = text or ""
    for src, dst in TW_TECH_LEXICON:
        t = t.replace(src, dst)
    # Clean spacing between CJK and Latin/Digits for display aesthetics
    t = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9$])", r"\1 \2", t)
    t = re.sub(r"([A-Za-z0-9%])([\u4e00-\u9fff])", r"\1 \2", t)
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
                    res.append("兩" + units[pos])
                elif d == 1 and pos == 1 and not res and val < 20:
                    res.append("十")
                else:
                    res.append(_DIGITS[d] + units[pos])
                zero_flag = False
        return "".join(res)

    if n < 10000:
        out = _section_4(n)
        return out
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
    Convert tech script line into 100% pure Mandarin-spoken form so F5-TTS pronounces
    every number, unit, price, percentage, and chip model naturally like `@hey_digitall`!
    """
    t = purify_tw_tech_text(raw_text)

    # 0. Brand names & display standards before digit conversion
    brand_pre_map = [
        (r"9\s*to\s*5\s*Mac", " Nine to Five Mac "),
        (r"\b3D\b", "三 D "),
        (r"\b4K\b", "四 K "),
        (r"\b5K\b", "五 K "),
        (r"\b6K\b", "六 K "),
        (r"\b8K\b", "八 K "),
        (r"Mini-?LED", " Mini L E D "),
        (r"Wi-?Fi\s*6E\b", " Wi-Fi 六 E "),
    ]
    for pat, repl in brand_pre_map:
        t = re.sub(pat, repl, t, flags=re.I)

    # 1. Years: 2020年..2035年 -> 二零二六年
    t = re.sub(r"\b(19\d\d|20\d\d)\s*年", lambda m: f"{year_to_zh(m.group(1))}年", t)

    # 2. Currencies: NT$34,900 / $1,299 / 1,299 USD / 899 美元
    t = re.sub(
        r"NT\$\s*(\d[\d,]*(?:\.\d+)?)",
        lambda m: f"新台幣{num_str_to_zh(m.group(1), two_as_liang=True)}元",
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

    # 3. Apple / Tech Chip Designators: M6 Pro -> M 六 Pro, M5 -> M 五, A20 -> A 二十, N1 -> N 一, Wi-Fi 7 -> Wi-Fi 七
    t = re.sub(
        r"\b([MANC])\s*(\d{1,2})(?=\s*(?:Pro|Max|Ultra|晶片|處理器|核心|版|\b))",
        lambda m: f"{m.group(1)} {int_to_zh(int(m.group(2)))}",
        t,
    )
    t = re.sub(
        r"(Wi-?Fi|藍牙|Bluetooth|Thunderbolt|USB)\s*(\d+)",
        lambda m: f"{m.group(1)} {int_to_zh(int(m.group(2)))}",
        t,
        flags=re.I,
    )

    # 4. Ranges: 4-5 年 / 13~15 吋 -> 四到五年
    t = re.sub(
        r"(\d+(?:\.\d+)?)\s*[-~～至]\s*(\d+(?:\.\d+)?)",
        lambda m: f"{num_str_to_zh(m.group(1))}到{num_str_to_zh(m.group(2))}",
        t,
    )

    # 5. Signed percentages & plain percentages: -10% -> 負百分之十, +12.5% -> 百分之十二點五, 12% -> 百分之十二
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

    # 6. Bandwidth, Storage, Frequency, Brightness, Power, Weight, Dimensions, Model Params (8B/14B/32B)
    unit_rules = [
        (r"(\d[\d,]*(?:\.\d+)?)\s*GB/s\b", lambda v: f"每秒{num_str_to_zh(v)} G 位元組"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*TB/s\b", lambda v: f"每秒{num_str_to_zh(v)} T 位元組"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*tok/s\b", lambda v: f"每秒{num_str_to_zh(v)}個 token "),
        (r"(\d[\d,]*(?:\.\d+)?)\s*GB\b", lambda v: f"{num_str_to_zh(v, two_as_liang=False)} G "),
        (r"(\d[\d,]*(?:\.\d+)?)\s*TB\b", lambda v: f"{num_str_to_zh(v, two_as_liang=True)} T "),
        (r"(\d[\d,]*(?:\.\d+)?)\s*Hz\b", lambda v: f"{num_str_to_zh(v)}赫茲"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*GHz\b", lambda v: f"{num_str_to_zh(v)} G 赫茲"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*(?:nits?|尼特)\b", lambda v: f"{num_str_to_zh(v, two_as_liang=True)}尼特"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*mAh\b", lambda v: f"{num_str_to_zh(v, two_as_liang=True)}毫安培小時"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*mm\b", lambda v: f"{num_str_to_zh(v)}公釐"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*kg\b", lambda v: f"{num_str_to_zh(v)}公斤"),
        (r"(\d[\d,]*(?:\.\d+)?)\s*W\b", lambda v: f"{num_str_to_zh(v)}瓦"),
        (r"(\d+(?:\.\d+)?)\s*B(?=\s*(?:模型|參數|\b))", lambda v: f"{num_str_to_zh(v)} B "),
    ]
    for pat, fn in unit_rules:
        t = re.sub(pat, lambda m, f=fn: f(m.group(1)), t, flags=re.I)

    # 7. Remaining standalone decimals & integers (e.g., 4,100 分 -> 四千一百分, 20 小時 -> 二十小時, 2 台 -> 兩台)
    t = re.sub(
        r"\b2(?=\s*(?:台|顆|款|種|倍|個|小時|年|天|支|張))",
        "兩",
        t,
    )
    t = re.sub(
        r"\d[\d,]*(?:\.\d+)?",
        lambda m: num_str_to_zh(m.group(0)),
        t,
    )

    # 8. Common Acronyms spoken cleanly in Mandarin tech channels (avoids F5-TTS "塞斯D" / "人PU" artifacts)
    acronym_map = [
        ("OLED", " O L E D "),
        ("SSD", "固態硬碟"),
        ("HDD", "傳統硬碟"),
        ("NVMe", "高速固態"),
        ("NPU", "神經網路引擎"),
        ("GPU", "繪圖核心"),
        ("CPU", "處理器核心"),
        ("RAM", "記憶體"),
        ("LLM", "大語言模型"),
        ("HDR", "高動態範圍"),
        ("HDMI", " H D M I "),
        ("SDXC", "記憶卡"),
        ("NAS", "網路儲存伺服器"),
        ("AI", " A I "),
        ("ProMotion", " Pro Motion "),
    ]
    for k, v in acronym_map:
        t = re.sub(rf"(?<![A-Za-z]){re.escape(k)}(?![A-Za-z])", v, t)

    t = re.sub(r"\s+", " ", t).strip()
    if t and t[-1] not in "。！？":
        t += "。"
    return t
