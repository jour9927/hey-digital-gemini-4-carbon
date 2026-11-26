# Hey Digital Gemini 4 Carbon 產線 V6 (`@hey_digitall`)

**1:1 原片級全自動無人值守長片 + 互動導流 Short 影片產線（含深度全文爬蟲、多關鍵字實拍鏡頭切分、`KeypointRCNN` 零人臉純硬體過濾、`SigLIP` 逐句旁白視覺語義精準對位、32B 查證產稿、台灣繁中純中文數字口播淨化、PyTorch 句級氣口聲紋克隆 + Whisper 時間軸對齊、Apple Metal `Three.js` WebGL2 真 3D 動態圖卡、Rhodes 電鋼琴閃避混音與 Telegram 自動交付）**

本專案逐幀逐秒對標 YouTube 科技深度頻道 **Hey Digital (`@hey_digitall`)** 全部參考長片與導流 Short，完整重建並升級外包剪輯師的全套製作鏈，實現從「自動選題 → 多源公開全文爬蟲與第三方硬體實拍池採集 → 32B 深度散文腳本與逐句 B-Roll / 3D 分鏡編導 → `KeypointRCNN` 零口播人臉過濾 + `SigLIP` 視覺語義對位 → Alan 本人 PyTorch 聲紋克隆與時間軸對齊 → 真 3D WebGL 動態圖卡與逐句實拍切鏡合成 → 長片、對應導流 Short 與縮圖自動傳送 Telegram」**全程 100% 無人值守（Unattended End-to-End）**。

---

## 七大工作站架構（V3 第三方素材逐句精準對位升級）

1. **深度全文公開爬蟲站 (`hey_digital_pipeline/crawler.py`)**
   - 支援 `MediaWiki explaintext=1` 多章節完整規格擷取、`9to5Mac` / `MacRumors` / `Ars Technica` / `The Verge` / `Hacker News` 正文與規格表深度抓取，以及 `Google News 台灣科技 RSS` 即時熱點掃描。
   - 所有文字與第三方實拍畫面來源皆綁定可驗證編號（`[SRC-01]` ~ `[SRC-20]`），同步呈現於 **左下角毛玻璃來源膠囊**、**動態瀏覽器引述視窗**、**片尾公開參考來源總表** 與 **YouTube 資訊欄**。

2. **多關鍵字第三方實拍切鏡、`KeypointRCNN` 零人臉過濾與 `SigLIP` 逐句對位站 (`hey_digital_pipeline/broll_matcher.py`)**
   - **逐句多關鍵字平行採集**：根據每鏡前半句與後半句旁白（如 `連接埠`、`120Hz 滑動`、`Mini-LED 亮度`、`多螢幕外接`、`時間軸剪片`、`SSD 測速`）平行啟動 14 組 `yt-dlp` 精準硬體檢索。
   - **無跳剪連續硬體特寫切分**：以 `2fps` RGB 縮圖差分偵測硬切點，精準擷取 `2.5s–5.0s` 單一連續運鏡特寫，自動剔除白底網頁、靜態簡報與底部燒錄外文字幕條。
   - **雙階段神經網路視覺把關**：
     1. **`KeypointRCNN-ResNet50-FPN` 三影格（`15% / 50% / 85%`）零人臉審查**：100% 剔除國外 YouTuber 棚拍口播人臉與子母畫面頭像，完整保留「雙手鍵盤打字／觸控板滑動」與「純硬體微距特寫」。
     2. **`google/siglip-base-patch16-384` 視覺語義對位**：優先於 `RTX Pro 6000` CUDA（斷線時無縫回退本機 `M3 Pro` Apple Metal MPS）計算每個純硬體鏡頭與當句旁白 `visual_desc` 的語義匹配分數，並強制全片鏡頭全域不重複。

3. **旗艦 32B 口語化編導產稿與 V4 零破音繁簡分詞口播淨化站 (`scriptwriter.py` + `cn_verbalizer.py`)**
   - 內嵌 `@hey_digitall` 原片逐字稿 Few-Shot 語氣錨點，強制產出真人面對面拆解風格的口語化播講稿，採用 **50% 逐句精準對位第三方硬體實拍 (`cinema_broll_video`) + 50% `Three.js` 3D 動態數據／選配模板** 黃金比例交錯編排。
   - **V6 視聽 1:1 逐字對齊口播淨化 (`cn_verbalizer.py`)**：
     - 螢幕字幕與口播維持 **1:1 詞序與詞彙一致**（保留 `MacBook Air M5` → `MacBook Air M五` 原順序，絕不暗中增減字；`SSD`／`RAM` 於顯示與口播端同步轉為 `固態硬碟`／`記憶體`）。
     - 口播端以純字級 `OpenCC('t2s')` 轉換並保留台灣科技詞彙，精準處理 `1TB` → `一 TB`（防連讀為 `ET`）、`153GB/s` → `每秒一百五十三G`、`120Hz` → `一百二十赫茲`、`20°C` → `二十度` 與 `拖著 zhe`。
   - **V6 中文詞法語法斷字與中文數字正規化字級對齊引擎 (`hey_digital_pipeline/subtitle_engine.py`)**：
     - 採用真實視覺字寬（1 中文字 = `1.0 em`、2 英數字 ≈ `1.0 em`）與嚴格單行排版（長片 `<= 21.8 em`、Short `<= 17.2 em`），並保護比例格式（如 `16:9`）不被半形冒號誤切。
     - 結合 `rjieba` / `jieba` 雙重分詞保底、科技複合詞鎖定（含 `表面溫差`、`溫控差距`、`擴充底座` 等）與動態規劃語法懲罰，**100% 根除破詞**與句尾懸掛虛詞。
     - 內建中文口語數字與英文規格單位雙向正規化骨架，搭配 Whisper `word_timestamps` 字級單調錨點對齊並持久化儲存 `whisper_words`，支援毫秒級離線重排字幕。

4. **多節點 GPU 算力調度與 V6 `Qwen3-TTS-1.7B` 雙母帶原聲級克隆站 (`worker/rtx6000_heavy_worker.py`)**
   - **本機優先安全與三層算力自動調度**：所有主腳本、設定檔、分鏡 JSON、克隆音檔與成片一律優先保存於本機（`Alan-Hub`）；配音與重活依序自動調度遠端 GPU Bridge → Tailscale `b850` (`RTX 5070 Ti` `/dev/shm` + `/tmp` 高速記憶體磁碟 CUDA，零干擾自動偵測顯存) → 本機 `M3 Pro` Apple Metal MPS。
   - **V6 `Qwen3-TTS-12Hz-1.7B-Base` 雙母帶零斷氣克隆**：採用 `Demucs (htdemucs)` 分離之零空白斷層長片母帶（`alan_v6_long_m5_24k.wav`）與輕快 Short 專屬母帶（`alan_v6_short_24k.wav`），搭配 `0.38s` 12Hz 編碼器句尾靜音邊界保護，徹底根除詞中斷氣與 `MacBook Air M5`、`120Hz ProMotion`、`Mini-LED`、`HDMI` 等中英混讀誤讀。
   - **Tailscale SSH 跨機叢集 (`config/ssh_config`)**：完整納管並驗證可達 `b850` / `b850-ubuntu` (`100.100.230.119`, RTX 5070 Ti 16GB)、`alan-fw` (`100.78.161.123`, AMD Ryzen AI Max+ 395 128GB) 與 `alan-mba` (`100.102.171.50`, M4 Air 24GB)。

5. **Apple Metal `Three.js` WebGL2 真 3D 動態圖卡與互動 UI 站 (`hey_digital_pipeline/motion_studio.py`)**
   - 保留並支援 6 大長片 3D / UI 動態母版：`webgl_memory_lanes`、`webgl_stacked_tco`、`webgl_ssd_particles`、`webgl_chip_die`、`webgl_dual_laptop`、`ui_configurator_crawl`。

6. **互動導流 Short 動態編排站 (`motion_studio.py` + `compositor.py`)**
   - `short_hook_dynamic`（直式純硬體動態實拍底 + 5 張彈簧貼紙）、`short_phone_interactive`（薄荷綠漸層 + iPhone 內嵌對位實拍畫面 + 觸控點擊波紋與動態 iOS 開關）、`short_broll_summary` 與 `cta_funnel_endcard`。

7. **Rhodes 電鋼琴閃避混音與 Telegram 一鍵交付站 (`audio_studio.py` + `tg_delivery.py`)**
   - 程序化合成 `86 BPM` Fender Rhodes 電鋼琴和弦進行 + 轉場／彈出／點擊音效 + FFmpeg `sidechaincompress` 閃避壓低與 `-14.2 LUFS` 響度標準化。
   - 支援 `--send-tg` 自動將完稿縮圖、長片（超過 50MB 自動壓製高畫質串流版）與導流 Short 推送至頻道經營者 Telegram。

---

## 快速啟動（無人值守一鍵成片 + Telegram 交付）

```bash
# 1. 完全無人值守產出長片 + 導流 Short + 縮圖並自動傳送至 Telegram
.venv/bin/python -u run_pipeline.py --video-type versus_guide --send-tg

# 2. 指定任意主題與影片類型（自動爬取公開規格、零人臉實拍 B-Roll 逐句對位與來源佐證）
.venv/bin/python -u run_pipeline.py \
  --topic "M4 Pro Mac mini 對決 AMD AI Max+ 395：本機跑大模型誰才不虧？" \
  --video-type ai_benchmark \
  --long-beats 28 \
  --short-steps 5 \
  --send-tg

# 3. 驗證 b850 與 Tailscale 節點 SSH 連通性
ssh -F config/ssh_config b850 "hostname && nvidia-smi -L"
```
