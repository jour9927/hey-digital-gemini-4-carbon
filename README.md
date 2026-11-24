# Hey Digital Gemini 4 Carbon 產線 (`@hey_digitall`)

**1:1 頻道級全自動無人值守長片 + 導流 Short 影片產線（含即時公開爬蟲、來源標註、PyTorch 聲紋克隆、AI 棚拍生圖與 Tailscale 分散式算力調度）**

本專案完整復刻並升級 YouTube 科技深度頻道 **Hey Digital (`@hey_digitall`)** 的外包剪輯師工作流，實現從「自動選題 → 多源公開資訊爬蟲與查證 → 附來源腳本與分鏡生成 → Alan 本人 PyTorch 聲音克隆配音 → AI 科技攝影棚實景與 3D 等距數據圖卡生成 → 長片與對應導流 Short 成片合成」**全程無人值守（Unattended End-to-End）**。

---

## 核心架構與頻道復刻亮點

1. **即時公開資訊爬蟲與雙重來源標註 (`hey_digital_pipeline/crawler.py`)**
   - 聚合 `9to5Mac`、`MacRumors`、`Ars Technica`、`The Verge`、`Hacker News Algolia API`、`Google News 台灣科技 RSS` 與 `Wikipedia REST API` 公開規格。
   - 自動比對 `config/channel_history.json` 排除已發佈重複題材，無人值守挑選最具選購反差張力的新題目。
   - 所有擷取數據皆綁定可追溯編號（`[SRC-01]` ~ `[SRC-12]`），同步渲染於 **畫面左下角毛玻璃來源膠囊**、**片尾公開資訊來源總表** 與 **YouTube 資訊欄參考連結**。

2. **多類型可重用 AI 編導引擎 (`hey_digital_pipeline/scriptwriter.py`)**
   - 支援五大類科技影片模板重複使用：
     - `versus_guide`：雙機 / 多陣營橫評選購對決（如 Air vs Pro、平板 vs 筆電）
     - `ai_benchmark`：本機 AI 算力、統一記憶體容量與頻寬實測避坑
     - `deep_review`：單品深度評測（硬體進化 vs 刀法妥協）
     - `hidden_tips`：新機到手必做設定與進階隱藏技巧
     - `news_analysis`：科技新品發布與定價邏輯深度拆解

3. **本機優先 + RTX Pro 6000 / Tailscale 叢集重活分流 (`worker/` + `remote_worker.py`)**
   - **安全原則**：所有最新主腳本、設定檔、分鏡 JSON 與成片資產一律優先保存於本機（`Alan-Hub`），遠端高速節點僅作無狀態算力執行，即使臨時節點斷線亦零資料遺失。
   - **RTX Pro 6000 重活引擎 (`worker/rtx6000_heavy_worker.py`)**：
     - **PyTorch 聲紋克隆**：採用 `Demucs (htdemucs)` 分離純淨乾聲 + `faster-whisper` 對齊聲紋提示詞 + `PyTorch F5-TTS` 零樣本流匹配克隆 Alan 頻道原聲（完美支援中英數科技術語混讀，自動正規化至 `-14.2 LUFS`）。
     - **AI 科技棚拍生圖**：支援 `FLUX.1-schnell` / `SDXL` 生成胡桃木格柵桌面、微距晶片與暗黑聚光燈硬體圖，並內建動態可用顯存偵測（`get_free_cuda_mb`），絕不干擾同節點其他並行工作階段。
   - **Tailscale SSH 跨機互連 (`config/ssh_config`)**：
     - 完整納管並驗證可達：`b850` / `b850-ubuntu` (`100.100.230.119`, RTX 5070 Ti 16GB)、`alan-fw` (`100.78.161.123`, AMD Ryzen AI Max+ 395 128GB)、`alan-mba` (`100.102.171.50`, M4 Air 24GB) 與 `alan-hub` (`100.119.215.91`, M3 Pro 36GB)。

4. **1:1 視覺母版與導流 Short 合成器 (`hey_digital_pipeline/compositor.py`)**
   - **長片 (`1920x1080@30fps`)**：深炭黑鏤空數字章節卡 (`chapter_dark_card`)、暖米灰 `#ECEAE4` 3D 等距晶片數據圖卡 (`studio_isometric_infographic`)、Ken Burns 推軌實景 + 浮動公開新聞網頁卡 (`cinema_tech_broll`)、深淺背景自適應無底框字幕。
   - **導流 Short (`1080x1920@30fps`)**：前 3 秒傾斜貼紙痛點鉤子 (`hook_sticker_broll`)、薄荷綠漸層手機框步驟實戰卡 (`step_phone_mockup`)、關聯長片縮圖導流收尾卡 (`cta_funnel_endcard`)。
   - **頻道縮圖 (`1920x1080`)**：深黑聚光燈底 + 純白與香檳金 (`#F0CA86`) 雙色超粗體標題。

---

## 快速啟動（無人值守一鍵成片）

```bash
# 1. 完全無人值守（自動爬蟲選題 -> 爬取公開來源 -> AI 產稿 -> RTX Pro 6000 克隆配音與生圖 -> 長片 + 導流 Short + 縮圖輸出）
python3 run_pipeline.py --video-type versus_guide

# 2. 指定任意主題與影片類型（同樣自動爬取公開規格與來源佐證）
python3 run_pipeline.py \
  --topic "M4 Pro Mac mini 對決 AMD AI Max+ 395：本機跑大模型誰才不虧？" \
  --video-type ai_benchmark \
  --long-beats 22 \
  --short-steps 5

# 3. 驗證 b850 與 Tailscale 節點 SSH 連通性
ssh -F config/ssh_config b850 "hostname && nvidia-smi -L"
```
