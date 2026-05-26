# 🏎️ F1 Pit Stop Prediction: EDA & Feature Engineering

本專案旨在針對 F1 賽車進站策略（Pit Stop Prediction）進行資料探索性分析（EDA）與特徵工程（Feature Engineering）。透過分析車手表現、輪胎耗損、單圈時間與賽事進度等歷史數據，我們萃取並建構出更具預測力的特徵，最終產出可直接用於機器學習模型訓練的乾淨數據集。

預測目標：**`PitNextLap`**（預測車手是否會在下一圈進站）。

---

## 🚀 專案亮點 (Key Features)

### 1. 📊 探索性資料分析 (Exploratory Data Analysis, EDA)
* **類別基數與分布**：可視化 `Compound`（輪胎配方）、`Race`（賽事）等類別變數的分佈狀況。
* **季前測試 vs 正賽對比**：使用 KDE Plot 比較 Pre-Season Testing 與實際比賽在輪胎壽命、單圈時間、衰退程度上的差異。
* **特徵相關性分析**：透過 Pearson 相關係數熱圖（Heatmap）找出與目標變數 `PitNextLap` 高度相關的數值特徵。
* **目標變數分群可視化**：繪製 Hexbin 與進階 KDE 密度圖，直觀展示在不同賽事進度下，進站與不進站的數據分布特徵。

### 2. 🔢 類別特徵編碼 (Categorical Encoding)
為了防止資料穿越（Data Leakage），所有編碼器皆僅在 `train` 集上進行擬合（Fit），並套用至 `train` 與 `test` 集：
* **Ordinal Encoding**：針對 `Compound`（輪胎配方）依照硬度進行有序編碼（`SOFT=0` 到 `WET` 最大）。
* **Label Encoding**：將 `Race`（賽事名稱）與 `Driver`（車手）轉換為模型可讀的整數索引，並處理測試集中的未知類別（填補為 `-1`）。

### 3. 🛠️ 特徵工程 (Feature Engineering)
根據 F1 賽事領域知識（Domain Knowledge），新增了多項衍生特徵以捕捉賽車動態變化：
* **輪胎狀態**：`TyreLife_per_Stint`, `TyreLife_squared`, `TyreLife_x_Compound`（輪胎壽命與配方的交互作用）。
* **單圈時間趨勢**：`LapTime_rolling3`（過去三圈平均圈速）, `LapTime_vs_rolling3`（當前圈速與平均的落差），皆依據 `Race` 與 `Stint` 進行分組計算以避免跨賽段污染。
* **賽事與位置進度**：`Laps_Since_LastPit`（距離上次進站圈數）, `Position_Change_cumsum`（賽段內名次累積變化）, `RaceProgress_squared`。
* **交互項特徵**：`TyreLife_x_RaceProgress`（賽事進度與輪胎壽命的綜合影響）。

---

## 📂 專案架構 (Project Structure)

```text
├── dataset/
│   ├── train.csv                 # 原始訓練集
│   ├── test.csv                  # 原始測試集
│   ├── train_engineered.csv      # 輸出：特徵工程後的訓練集
│   └── test_engineered.csv       # 輸出：特徵工程後的測試集
├── eda_feature_engineering.py    # 主要執行的 Python 腳本
└── README.md                     # 專案說明文件
