# 🚀 Hybrid Trading System for Taiwan Stock Index (^TWII)

這是一個先進的演算法交易系統，結合了用於價格預測的 **LSTM-SSAM** (Long Short-Term Memory with Sequential Self-Attention) 以及用於交易決策的 **Pro Trader RL** (Reinforcement Learning)。

## ✨ 核心特色 (Key Features)

| 特色 | 說明 |
|---------|-------------|
| **本地資料整合** | TWII 歷史資料採本地 CSV 管理 (`twii_data_from_2000_01_01.csv`)，確保成交量單位 (億元) 正確，並具備自動更新機制 |
| **嚴謹訓練流程** | **Data Leakage Prevention**: LSTM 模型訓練時的資料縮放 (Scaling) 嚴格限制在訓練集內，防止 Look-ahead Bias |
| **LSTM-SSAM 預測** | T+1 與 T+5 價格預測，並使用 MC Dropout 進行不確定性估計 |
| **遷移學習 (Transfer Learning)** | 使用全球指數進行預訓練 (Pre-train) → 針對 ^TWII 進行微調 (Fine-tune) |
| **特徵融合 (Feature Fusion)** | 整合 24 種特徵，包含 LSTM 預測值與信心分數 (T+1/T+5) |
| **PPO Agent** | 分離的買入 (Buy) 與賣出 (Sell) 代理人，並具備類別平衡機制 |
| **回測 (Backtesting)** | 完整的模擬回測，包含停損機制與績效指標計算 |

## 📊 績效結果 (2023-Present)

| 指標 (Metric) | V4 Standard | V4 Rolling LSTM | Buy & Hold |
|--------|------:|------:|------:|
| **總報酬率 (ROI)** | 85.49% | **104.06%** | 98.24% |
| **年化報酬率** | 23.53% | **27.43%** | 26.18% |
| **夏普值 (Sharpe)** | 1.47 | 1.45 | 1.21 |
| **最大回撤** | -17.23% | -23.49% | -28.69% |
| **勝率** | 100% (5 次) | 62.5% (8 次) | N/A |

## 🏗️ 系統架構 (Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                     HYBRID TRADING SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  LSTM T+1    │    │  LSTM T+5    │    │    技術指標       │  │
│  │   預測模型    │    │  + MC Dropout│    │  (Indicators)    │  │
│  └──────┬───────┘    └──────┬───────┘    └────────┬─────────┘  │
│         │                   │                      │            │
│         └───────────────────┼──────────────────────┘            │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │    23 特徵融合   │                          │
│                    │  (Feature Fusion)│                         │
│                    └────────┬────────┘                          │
│                             │                                    │
│         ┌───────────────────┴───────────────────┐               │
│         │                                       │               │
│  ┌──────▼──────┐                        ┌──────▼──────┐        │
│  │  Buy Agent  │                        │  Sell Agent │        │
│  │    (PPO)    │                        │    (PPO)    │        │
│  └──────┬──────┘                        └──────┬──────┘        │
│         │                                      │                │
│         └──────────────────┬───────────────────┘                │
│                            │                                     │
│                   ┌────────▼────────┐                           │
│                   │    交易訊號      │                           │
│                   └─────────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 專案結構 (Project Structure)

```
hybrid-trader-v04/
├── ptrl_hybrid_system.py           # 核心系統 (資料載入/特徵計算/訓練邏輯)
├── update_twii_data.py             # 資料更新腳本 (自動抓取最新 TWII 數據)
├── twii_data_from_2000_01_01.csv   # 本地 TWII 歷史資料庫 (Volume: 億元)
│
├── # === 訓練腳本 ===
├── train_lstm_models.py            # LSTM 訓練 (固定日期範圍)
├── train_v3_models.py              # V3 訓練 (輕量化微調)
├── train_v4_models.py              # V4 訓練 (標準微調)
├── train_v4_rolling_lstm.py        # V4 Rolling LSTM 訓練 (年度滾動) ✨ NEW
│
├── # === 每日維運 ===
├── daily_ops_dual.py               # 每日維運 - V3+V4 雙策略
├── daily_ops_v4_rolling.py         # 每日維運 - Rolling LSTM V4 ✨ NEW
├── daily_ops_dual_intraday.py      # 盤中即時分析
│
├── # === 回測腳本 ===
├── backtest_v3_no_filter.py        # V3 回測
├── backtest_v4_no_filter.py        # V4 回測
├── backtest_v4_rolling_lstm.py     # V4 Rolling LSTM 回測 ✨ NEW
├── backtest_v4_dca_hybrid_no_filter.py # DCA 混合回測
│
├── # === 模型目錄 ===
├── models_hybrid_v3/               # V3 RL 模型
├── models_hybrid_v4/               # V4 RL 模型
├── models_hybrid_v4_rolling/       # V4 Rolling LSTM RL 模型 ✨ NEW
└── rolling_lstm_models/            # Rolling LSTM 模型 (按年份存放) ✨ NEW
    ├── 2006/                       # 2006 年用的 LSTM 模型
    ├── 2007/
    ├── ...
    └── 2022/                       # 2022 年用的 LSTM 模型 (用於推論)
```

## 🛠️ 安裝說明 (Installation)

### 建議使用虛擬環境 (Virtual Environment)
在 Windows 上使用虛擬環境可以避免套件版本衝突，強烈建議使用。

**方法一：使用自動腳本 (推薦)**
```powershell
.\setup_env.ps1
```

**方法二：手動設定**
```powershell
# 1. 建立虛擬環境
python -m venv venv

# 2. 啟動虛擬環境
.\venv\Scripts\Activate.ps1

# 3. 安裝套件
pip install -r requirements.txt
```

### ⚡ GPU 加速設定 (重要)
本專案建議使用 NVIDIA 顯卡進行訓練加速。

**方法一：使用 setup_env.ps1 (自動)**
腳本會自動安裝支援 CUDA 11.8 的 PyTorch 版本。

**方法二：手動安裝**
若您手動執行 `pip install -r requirements.txt`，預設會安裝 CPU 版本。請執行以下指令將其替換為 GPU 版本：

```powershell
# 1. 移除 CPU 版本
pip uninstall torch torchvision torchaudio -y

# 2. 安裝 GPU 版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 系統需求 (Dependencies)

```
tensorflow>=2.10
stable-baselines3>=2.0
gymnasium
yfinance
pandas
numpy
ta
torch
tqdm
matplotlib
psutil
```

## 🚀 快速開始 (Quick Start)

### 1. 訓練 LSTM 模型 (長週期)

```bash
python train_lstm_models.py
```

### 2. 訓練 RL 模型 (V3 vs V4)

本專案提供兩個版本的 RL 訓練腳本，請依需求選擇：

| 特性 | V3 (Lightweight) | V4 (Standard) |
|------|------------------|---------------|
| **用途** | 輕量版，適合快速實驗 | 標準版，適合完整訓練 |
| **Buy Fine-tune** | 200,000 步 | 1,000,000 步 |
| **Sell Fine-tune** | 100,000 步 | 300,000 步 |
| **指令** | `python train_v3_models.py` | `python train_v4_models.py` |
| **輸出目錄** | `models_hybrid_v3/` | `models_hybrid_v4/` |

### 3. 每日維運 (Daily Operations)

自動化腳本能完成「LSTM 重訓 → 特徵工程 → RL 推論 → 報告生成」全流程。

#### 📌 盤後分析 (收盤後執行)

- **雙策略比較 (推薦)**: 同時執行 V3 與 V4 模型並產生綜合建議。
  ```bash
  python daily_ops_dual.py
  ```

- **單策略運行**:
  ```bash
  python daily_ops_v3.py  # 僅 V3
  python daily_ops_v4.py  # 僅 V4
  ```

**輸出目錄**: `daily_runs/YYYY-MM-DD/`

#### ⏱️ 盤中即時分析 (交易時段內執行)

```bash
python daily_ops_dual_intraday.py
```

**流程說明：**
1. 從**證交所盤中 API** (`mis.twse.com.tw`) 下載當日即時 OHLC *(2025-12-12 更新)*
2. 使用 CSV 前 5 日成交量平均作為當日預估成交量
3. 使用上述資料完整訓練 LSTM 模型 (T+5 及 T+1)
4. 進行 V3/V4 雙策略推論並輸出報告

**特點：**
- ✅ **即時資料**：使用證交所 API 取得真正的盤中價格（yfinance 有 15+ 分鐘延遲）
- ✅ 獨立運作，**不需先執行 daily_ops_dual.py**
- ✅ **不修改原始 CSV** (使用 CSV 交換策略：備份 → 訓練 → 恢復)
- ✅ 輸出到獨立資料夾 `intraday_runs/YYYY-MM-DD_HHMMSS/`
- ⏰ 執行時間約 20-40 分鐘 (LSTM 訓練時間)

**輸出目錄結構：**
```
intraday_runs/2025-12-11_120731/
├── lstm_models/          # 盤中訓練的 LSTM 模型
├── cache/                # 暫存資料
└── reports/
    ├── intraday_summary.txt
    └── intraday_summary.json
```

---

**功能特點 (v2.7)：**
- **全時推論模式**: 無論 Donchian 濾網狀態，AI 都會執行預測並顯示意圖
- **濾網狀態標記**: `BUY`, `WAIT`, `FILTERED (AI: BUY)`, `FILTERED (AI: WAIT)`
- **情境分析**: Sell Agent 針對三種持倉情境 (成本區/獲利+10%/虧損-5%) 提供建議
- **數據匯出**: 自動匯出 `raw_data.csv` 和 `processed_features.csv` 供除錯檢查
- **統一訓練天數**: 三個 daily_ops 腳本均使用 T+5/2200天, T+1/2000天
- 輸出 JSON 與 TXT 戰情報告

### 4. 無濾網回測 (No-Filter Backtest)

測試 AI 在每天都可進場 (無 Donchian 濾網限制) 的情況下的績效：

```bash
python backtest_v3_no_filter.py  # V3 無濾網回測
python backtest_v4_no_filter.py  # V4 無濾網回測
```

**功能特點 (v2.9)：**

| 功能 | 說明 |
|------|------|
| **自訂日期範圍** | 透過 `--start` 和 `--end` 參數指定回測期間 |
| **動態檔名** | 輸出檔案自動包含日期範圍，避免覆蓋 |
| **Benchmark 比較** | 策略績效 vs Buy & Hold 並排顯示 |
| **績效摘要** | 控制台、圖表、CSV 三處同步顯示 |

**使用範例：**
```bash
# 預設日期 (2023-01-01 至今)
python backtest_v4_no_filter.py

# 自訂開始日期
python backtest_v4_no_filter.py --start 2020-01-01

# 自訂完整日期範圍
python backtest_v4_no_filter.py --start 2015-01-01 --end 2023-12-31
```

### 5. DCA + AI 混合策略回測

測試「定期定額 + AI 自由操作」混合策略的績效：

```bash
python backtest_v4_dca_hybrid_no_filter.py
```

**策略說明：**
- 每年年初獲得 60 萬新資金
- 50% 分 12 個月定期定額投入 (買入後不賣)
- 50% 由 AI 自由決定買賣時機

**比較基準：**
1. 純定期定額：每月 5 萬元
2. 年初一次投入：每年 60 萬 Buy & Hold

**輸出檔案：**
```
results_backtest_v4_dca_hybrid_no_filter/
├── backtest_v4_dca_hybrid_no_filter_20240102_20251205.png
├── metrics_v4_dca_hybrid_no_filter_20240102_20251205.csv
└── trades_v4_dca_hybrid_no_filter_20240102_20251205.csv
```

### 🔍 回測腳本功能比較

| 功能 | `backtest_v3_no_filter.py` | `backtest_v4_no_filter.py` | `backtest_v4_dca_hybrid_no_filter.py` |
|------|:---:|:---:|:---:|
| 自訂日期範圍 | ✅ | ✅ | ✅ |
| 動態檔名 | ✅ | ✅ | ✅ |
| Benchmark 比較 | ✅ | ✅ | ✅ |
| DCA + AI 混合策略 | ❌ | ❌ | ✅ |
| **LSTM 模型日期篩選** | ✅ (v3.0) | ✅ (v3.0) | ✅ |

> [!IMPORTANT]
> **LSTM 模型日期篩選 (v3.0 更新)**：所有回測腳本現在都會根據回測 `start_date` 來選擇 LSTM 模型，確保只使用 `train_end < start_date` 的模型，避免資料洩漏 (look-ahead bias)。

## 📈 訓練流程 (Training Pipeline)

### Phase 1: 數據整合 (Unified Data Source)
- **本地數據**: ^TWII 使用本地 `twii_data_from_2000_01_01.csv`，確保成交量單位正確 (億元)。
- **自動更新**: 系統自動檢查並透過 `update_twii_data.py` 補齊最新交易日資料。
- **國際指數**: 下載 4 個全球指數：^GSPC, ^IXIC, ^SOX, ^DJI (from yfinance)
- **影響範圍**: 涵蓋 V3/V4 訓練、所有回測腳本以及每日維運腳本 (Daily Ops)。

### Phase 2: 特徵工程 (Feature Engineering)
- 包含 24 種特徵 (v3.0 更新)：
  - 標準化 OHLC 價格
  - 唐奇安通道 (Donchian Channel)、超級趨勢 (SuperTrend)
  - 平均K線 (Heikin-Ashi) 型態
  - RSI, MFI, ATR 指標
  - 相對強度 (Relative Strength) 指標
  - **LSTM_Pred_1d**: T+1 預測漲幅
  - **LSTM_Conf_1d**: T+1 信心度 (MC Dropout) ✨ NEW
  - **LSTM_Pred_5d**: T+5 預測漲幅
  - **LSTM_Conf_5d**: T+5 信心度 (MC Dropout)

### Phase 3: 預訓練 (Pre-training)
- Buy Agent: 1,000,000 步 (類別平衡採樣)
- Sell Agent: 500,000 步

### Phase 4: 微調與回測 (Fine-tuning & Backtesting)
- 微調：針對 ^TWII (2000-2022) 進行訓練，Learning Rate = 1e-5
- 回測：驗證數據集 (2023-Present)

### ⚠️ 資料紀律 (Data Discipline)

> [!IMPORTANT]
> **防止資料洩漏 (Data Leakage Prevention)**
> 
> 本系統採用嚴格的時間切分策略，確保模型在訓練時不會看到驗證期的資料。

| 階段 | 資料範圍 | 說明 |
|------|----------|------|
| **LSTM 訓練** | 2000-01-01 ~ 2022-12-31 | 使用 `train_lstm_models.py` 設定的 `TRAIN_END` |
| **RL 預訓練** | 2000-01-01 ~ 2022-12-31 | 全球指數 (^TWII, ^GSPC, ^IXIC, ^SOX, ^DJI) 截止於 `SPLIT_DATE` |
| **RL 微調** | ^TWII < 2023-01-01 | 只用 `SPLIT_DATE` 之前的 TWII 資料 |
| **RL 驗證/回測** | ^TWII >= 2023-01-01 | 模型完全沒見過的資料 |

**關鍵設定 (2025-12-11 更新)：**
```python
# train_lstm_models.py
TRAIN_END = "2022-12-31"

# train_v3_models.py / train_v4_models.py
SPLIT_DATE = '2023-01-01'
raw_data = hybrid.fetch_index_data(DATA_PATH, start_date="2000-01-01", end_date=SPLIT_DATE)
```

**時間線視覺化：**
```
LSTM 訓練期:      2000 ─────────────────────── 2022-12-31
RL 訓練/微調期:   2000 ─────────────────────── 2022-12-31
                                                     │
RL 驗證/回測期:                                2023-01-01 ─────── 今天
                                               (模型未見過)
```

### Phase 5: 訓練監控 (Training Monitoring)
本系統整合了 **TensorBoard** 進行訓練過程的即時監控。

**自動記錄的指標：**
- `rollout/ep_rew_mean`: 平均獎勵
- `train/loss`: 總損失
- `train/policy_gradient_loss`: 策略梯度損失
- `train/value_loss`: 價值函數損失
- `train/entropy_loss`: 熵損失
- `eval/mean_reward`: 驗證集平均獎勵 (EvalCallback)

**如何使用 TensorBoard：**
```powershell
# 在專案目錄下執行
tensorboard --logdir ./tensorboard_logs/

# 然後開啟瀏覽器前往
# http://localhost:6006
```

**日誌存放位置：**
- `./tensorboard_logs/`: TensorBoard 日誌
- `./logs/`: EvalCallback 評估結果
- `models_hybrid/best_tuned/`: 驗證集最佳模型

---

## 📊 輸出結果 (Output)

執行 `ptrl_hybrid_system.py` 後，您將獲得：

- `models_hybrid/ppo_buy_twii_final.zip`: 微調後的 Buy Model
- `models_hybrid/ppo_sell_twii_final.zip`: 微調後的 Sell Model
- `results_hybrid/final_performance.png`: 績效圖表
- `tensorboard_logs/`: 訓練過程日誌 (可用 TensorBoard 查看)

## 🔧 V3 vs V4 版本比較

| 項目 | V3 (Lightweight) | V4 (Standard) | 原始版 (ptrl_hybrid_system.py) |
|-----|------------------|-----------------|--------------------------------|
| **Pre-train Buy** | 1,000,000 | 1,000,000 | 1,000,000 |
| **Pre-train Sell** | 500,000 | 500,000 | 500,000 |
| **Fine-tune Buy** | **200,000** | **1,000,000** | 1,000,000 |
| **Fine-tune Sell** | **100,000** | **300,000** | 300,000 |
| **信心度門檻** | [0.001, 0.010] v2.5 | [0.001, 0.010] v2.5 | [0.005, 0.015] (舊版) |
| **特徵快取** | 強制清除 | 強制清除 | 使用快取 (需手動清除) |
| **模型路徑** | `models_hybrid_v3` | `models_hybrid_v4` | `models_hybrid` |

---

## 🔄 V4 Rolling LSTM 策略 (NEW - 2025-12-12)

### 設計理念

傳統訓練方式中，RL Agent 會看到整個訓練期間的 LSTM 預測。但 LSTM 是用**全部歷史資料**訓練的，這可能導致 RL Agent 過度依賴「未來資訊」。

**Rolling LSTM Strategy** 解決了這個問題：
- 每年訓練一個新的 LSTM 模型，只使用**該年之前**的資料
- RL Agent 在 2010 年訓練時，使用的是「只看過 2009 年底前資料」的 LSTM 模型
- 確保 LSTM 預測永遠是 **out-of-sample**

### 訓練時間線

```
年份  LSTM 訓練資料          RL 訓練資料
2006  2000 ~ 2005-12-31  →  2006 年
2007  2000 ~ 2006-12-31  →  2007 年
...
2022  2000 ~ 2021-12-31  →  2022 年 → 最終模型
```

### 快速開始

```bash
# 1. 完整訓練 (2006-2022, 約 1.5 小時)
python train_v4_rolling_lstm.py

# 2. 回測驗證
python backtest_v4_rolling_lstm.py

# 3. 每日維運 (使用 2022 年訓練完成的模型)
python daily_ops_v4_rolling.py
```

### 參數說明

```bash
# Dry Run (只看訓練計畫)
python train_v4_rolling_lstm.py --dry-run

# 指定年份範圍
python train_v4_rolling_lstm.py --start-year 2020 --end-year 2022

# 只訓練 LSTM (不訓練 RL)
python train_v4_rolling_lstm.py --lstm-only
```

### 輸出結構

```
rolling_lstm_models/
├── 2006/
│   ├── saved_models_5d/      # T+5 LSTM 模型
│   └── saved_models_multivariate/  # T+1 LSTM 模型
├── 2007/
├── ...
└── 2022/                      # 用於日常推論的版本

models_hybrid_v4_rolling/
├── ppo_buy_2006.zip ~ ppo_buy_2022.zip  # 各年度 Buy Agent
├── ppo_sell_2006.zip ~ ppo_sell_2022.zip  # 各年度 Sell Agent
├── ppo_buy_twii_final.zip     # 最終 Buy Agent (= 2022)
└── ppo_sell_twii_final.zip    # 最終 Sell Agent (= 2022)

results_backtest_v4_rolling/
├── backtest_v4_rolling_*.png  # 績效圖表
├── metrics_v4_rolling_*.csv   # 績效指標
├── trades_v4_rolling_*.csv    # 交易明細
└── daily_confidence_v4_rolling_*.csv  # 每日信心度
```

### V4 Standard vs V4 Rolling LSTM

| 項目 | V4 Standard | V4 Rolling LSTM |
|------|-------------|-----------------|
| **LSTM 訓練方式** | 一次性 (2000-2022) | 年度滾動更新 |
| **RL 訓練方式** | 一次性看所有資料 | 每年微調一次 |
| **訓練時間** | ~30 分鐘 | ~1.5 小時 |
| **LSTM 預測** | 可能含 in-sample | 保證 out-of-sample |
| **每日維運** | 需每日重訓 LSTM | 使用固定 LSTM (~30 秒) |
| **模型更新** | 較複雜 | 每年底執行一次 |

### 績效比較 (2023-01-03 ~ 2025-12-12)

| 指標 | V4 Standard | V4 Rolling LSTM |
|------|-------------|-----------------|
| 總報酬率 | 85.49% | **104.06%** |
| 年化報酬 | 23.53% | **27.43%** |
| 夏普值 | **1.47** | 1.45 |
| 最大回撤 | **-17.23%** | -23.49% |
| 交易次數 | 5 次 | 8 次 |
| 勝率 | **100%** | 62.5% |

> **觀察**：Rolling LSTM 策略報酬更高但風險也較大。適合願意承受更大波動以追求更高報酬的投資者。

---

## 🔮 LSTM 信心度解讀指南 (Confidence Interpretation)

### 計算原理 (Methodology)
信心度 (`LSTM_Conf_1d`, `LSTM_Conf_5d`) 是基於 **蒙地卡羅 Dropout (MC Dropout)** 計算的：
1. 對同一筆資料進行 30 次預測（每次 Dropout 隨機遮蔽不同神經元）
2. 計算這 30 次預測的**變異係數 (CV)** = 標準差 ÷ 平均值
3. CV 越小 → 模型越穩定 → 信心度越高

### 門檻設定 (v3.0)

| 模型 | threshold_high | threshold_low | 說明 |
|------|----------------|---------------|------|
| **T+1** | 0.008 (0.8%) | 0.040 (4.0%) | 範圍較寬，適應較高的 CV 分佈 |
| **T+5** | 0.001 (0.1%) | 0.010 (1.0%) | 範圍較窄，模型本身較穩定 |

```python
# ptrl_hybrid_system.py add_lstm_features()
# T+1 信心度
score_1d = 1.0 - (cv - 0.008) / (0.040 - 0.008)
conf_1d = np.clip(score_1d, 0.0, 1.0)

# T+5 信心度
score_5d = 1.0 - (cv - 0.001) / (0.010 - 0.001)
conf_5d = np.clip(score_5d, 0.0, 1.0)
```

### 分數對照表

| 信心度 | 解讀 | 建議 |
|--------|------|------|
| **0.8+** | 🟢 **高信心** - 模型非常確定 | 預測可靠度高，可作為主要參考 |
| **0.6-0.8** | 🟡 **中等偏高** - 正常水準 | 預測可參考，但需結合其他指標 |
| **0.4-0.6** | 🟡 **中等** - 略有不確定性 | 預測僅供輔助參考 |
| **< 0.4** | 🔴 **低信心** - 模型不確定 | 預測不穩定，謹慎採信 |

### 實際應用建議
1. **信心度 0.8+**：可以更積極地參考 LSTM 的漲跌預測
2. **信心度 0.6-0.8**：預測方向可參考，但點位預估需打折扣
3. **信心度 0.4-0.6**：預測僅供輔助參考，建議搭配其他技術指標
4. **信心度 < 0.4**：模型對當天的判斷較不確定，可能是因為市場處於異常波動期

---

## 📚 參考文獻 (References)

- **Pro Trader RL**: [Paper Implementation](https://arxiv.org/abs/xxxx)
- **LSTM-SSAM**: Sequential Self-Attention for time series prediction
- **MC Dropout**: Uncertainty estimation via Monte Carlo Dropout

## 📄 授權 (License)

MIT License

## 👤 作者 (Author)

Phil Liang

---

*Built with Python, TensorFlow, Stable-Baselines3, and ❤️*
