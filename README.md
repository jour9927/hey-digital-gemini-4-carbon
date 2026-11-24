# Hey Digital Gemini 4 Carbon 產線 V2 (`@hey_digitall`)

**1:1 原片級全自動無人值守長片 + 互動導流 Short 影片產線（含深度全文爬蟲、實拍 B-Roll 自動採集、32B 查證產稿、台灣繁中純中文數字口播淨化、PyTorch 句級氣口聲紋克隆 + Whisper 時間軸對齊、Apple Metal `Three.js` WebGL2 真 3D 動態圖卡、`LTX-Video` I2V AI 運鏡與 Rhodes 電鋼琴閃避混音）**

本專案逐幀逐秒對標 YouTube 科技深度頻道 **Hey Digital (`@hey_digitall`)** 全部參考長片與導流 Short，完整重建並升級外包剪輯師的全套製作鏈，實現從「自動選題 → 多源公開全文爬蟲與實拍 B-Roll 採集 → 32B 深度散文腳本與 3D 分鏡導演 → Alan 本人 PyTorch 聲紋克隆與時間軸對齊 → 真 3D WebGL 動態圖卡與 AI 動態攝影 → 長片與對應導流 Short 成片混音輸出」**全程 100% 無人值守（Unattended End-to-End）**。

---

## 六大工作站架構（V2 全站升級）

1. **深度全文公開爬蟲與實拍 B-Roll 自動採集站 (`hey_digital_pipeline/crawler.py`)**
   - 支援 `MediaWiki explaintext=1` 多章節完整規格擷取、`9to5Mac` / `MacRumors` / `Ars Technica` / `The Verge` / `Hacker News` 正文與規格表深度抓取，以及 `Google News 台灣科技 RSS` 即時熱點掃描。
   - 內建 `harvest_real_broll_clips()`：自動透過 `yt-dlp --download-sections` 採集公開產品發表與上手實拍動態畫面，經亮度對比過濾裁切為乾淨 `1920x1080@30fps` 實拍 B-Roll 鏡頭。
   - 所有文字與畫面來源皆綁定可驗證編號（`[SRC-01]` ~ `[SRC-12]`），同步呈現於 **左下角毛玻璃來源膠囊**、**動態瀏覽器引述視窗**、**片尾公開參考來源總表** 與 **YouTube 資訊欄**。

2. **旗艦 32B 編導產稿與台灣繁中口播淨化站 (`scriptwriter.py` + `cn_verbalizer.py`)**
   - 內嵌 `@hey_digitall` 原片逐字稿 Few-Shot 語氣錨點，優先調度 `RTX Pro 6000` 上的 `Qwen/Qwen2.5-32B-Instruct`（支援 `Gemini 2.5 Pro` / 本機 `Ollama` 自動回退），產出 **27~28 鏡高密度長片分鏡** 與 **8 鏡互動導流 Short 分鏡**。
   - 支援五大可重用影片類型：`versus_guide`、`ai_benchmark`、`deep_review`、`hidden_tips`、`news_analysis`。
   - `cn_verbalizer.py` 自動將 80+ 項兩岸科技術語統一為台灣繁中慣用語（記憶體、硬碟、晶片、螢幕、影片等），並將所有金額、晶片型號、頻寬與百分比（如 `$1,299`、`M6 Pro`、`170 GB/s`）100% 轉寫為自然中文讀法 `tts_text`，徹底根除 TTS 英文唸數字與吞字問題。

3. **RTX Pro 6000 重活算力與句級氣口 PyTorch 聲紋克隆站 (`worker/rtx6000_heavy_worker.py`)**
   - **本機優先安全原則**：所有最新主腳本、設定檔、分鏡 JSON 與成片資產一律優先保存於本機（`Alan-Hub`），遠端節點透過 `/dev/shm` 高速記憶體磁碟無狀態執行，臨時節點斷線亦零資料遺失。
   - **PyTorch 聲紋克隆與 Whisper 對齊**：以 `Demucs (htdemucs)` 分離 Alan 純淨人聲乾音，透過 `PyTorch F5-TTS` 分句合成並保留自然 `0.27s` 換氣停頓與完整尾音收束；隨後在 CUDA 上執行 `openai/whisper-large-v3-turbo` 驗證語音完整度並輸出毫秒級字幕時間軸 `caption_timings`。
   - **AI 商業棚拍與 I2V 動態運鏡**：採用 `RealVisXL_V4.0` 生成高質感硬體棚拍底圖，並以 `Lightricks/LTX-Video` 渲染真實視差運鏡動態影片。
   - **Tailscale SSH 跨機叢集 (`config/ssh_config`)**：完整納管並驗證可達 `b850` / `b850-ubuntu` (`100.100.230.119`, RTX 5070 Ti 16GB)、`alan-fw` (`100.78.161.123`, AMD Ryzen AI Max+ 395 128GB) 與 `alan-mba` (`100.102.171.50`, M4 Air 24GB)。

4. **Apple Metal `Three.js` WebGL2 真 3D 動態圖卡與互動 UI 站 (`hey_digital_pipeline/motion_studio.py`)**
   - 透過 Playwright 驅動本機 Chrome Metal WebGL2 與 `Three.js` (`PCFSoftShadowMap` 真實柔光陰影 + 三次貝茲 3D 運鏡 + 彈簧物理)，支援 6 大長片動態母版：
     - `webgl_memory_lanes`：雙軌 / 三軌 3D 記憶體與頻寬方塊通道 + 巨型細體數字動態計數與差值膠囊
     - `webgl_stacked_tco`：3D 鋁合金筆電推鏡至雙底座加配價格堆疊柱（重現入門加價逼近高階款）
     - `webgl_ssd_particles`：3D NAND 快閃晶圓陣列 + 拋物線資料方塊飛入 3D 筆電
     - `webgl_chip_die`：3D SoC 晶片封裝環繞運鏡與發光 CPU / GPU / NPU 核心矩陣
     - `webgl_dual_laptop`：3D 雙機並排高負載競速與熱力環 vs 冷卻氣流環對比
     - `ui_configurator_crawl`：Apple 風格選配器自動滑鼠點擊波紋 + 真實爬蟲報導螢光筆劃重點視窗

5. **互動導流 Short 動態編排站 (`motion_studio.py` + `compositor.py`)**
   - `short_hook_dynamic`：直式動態 B-Roll 背景 + 5 張圓角貼紙與軋型放大鏡圖示依序彈簧彈出。
   - `short_phone_interactive`：薄荷綠漸層底 + 全尺寸 iPhone 邊框 + 4 列動態 iOS 規格卡片（觸控游標滑動點擊波紋、綠色開關自動切換、黃色聚焦框與分段進度條），底部描邊字幕零重疊。
   - `cta_funnel_endcard`：關聯長片縮圖動態推鏡與導流按鈕。

6. **Rhodes 電鋼琴律動配樂、轉場音效與閃避混音站 (`hey_digital_pipeline/audio_studio.py`)**
   - 程序化合成 `86 BPM` Fender Rhodes 電鋼琴爵士和弦進行（`Cmaj9 → Am11 → Dm9 → G13sus4`），於 `1.85s` 無口播章節黑卡自然浮現，口播進入時透過 FFmpeg `sidechaincompress` 平滑閃避壓低，並精準疊加章節氣流轉場音（`whoosh`）、貼紙彈出音（`pop`）與介面點擊音（`click`），最終母帶標準化至 `-14.2 LUFS`。

---

## 快速啟動（無人值守一鍵成片）

```bash
# 1. 完全無人值守（自動爬蟲選題 -> 深度全文與實拍 B-Roll 採集 -> 32B 產稿 -> RTX Pro 6000 克隆配音與生圖/I2V -> 3D WebGL 長片 + 互動 Short + 縮圖輸出）
python3 -u run_pipeline.py --video-type versus_guide

# 2. 指定任意主題與影片類型（同樣自動爬取公開規格、實拍 B-Roll 與來源佐證）
python3 -u run_pipeline.py \
  --topic "M4 Pro Mac mini 對決 AMD AI Max+ 395：本機跑大模型誰才不虧？" \
  --video-type ai_benchmark \
  --long-beats 28 \
  --short-steps 5

# 3. 驗證 b850 與 Tailscale 節點 SSH 連通性
ssh -F config/ssh_config b850 "hostname && nvidia-smi -L"
```
