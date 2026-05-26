<<<<<<< HEAD
# =============================================================================
# F1 Pit Stop Prediction — EDA + Feature Engineering (資料探索與特徵工程)
# =============================================================================

# ── 0. Imports ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# ── 1. Load Data ──────────────────────────────────────────────────────────────
# 載入訓練集與測試集
train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')

# 基礎資料探索：檢視資料型態、統計數值與缺失值狀況
train.info()
print(train.describe())
print("Train Nulls:\n", train.isnull().sum())
print("Test Nulls:\n", test.isnull().sum())

# 分離欄位類型（以 train 為基準）
# 自動將欄位分為類別型 (Categorical) 與數值型 (Numerical)，方便後續批次處理
cat_cols = train.select_dtypes(exclude=['number']).columns.to_list()
num_cols = train.select_dtypes(include=['number']).columns.to_list()
print(f"\nCategorical Columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical Columns   ({len(num_cols)}): {num_cols}")


# ── 2.1 類別欄位基數 (Cardinality) ──────────────────────────────────────────────
# 觀察類別欄位中有多少種不同的獨立值 (Unique values)，評估是否會有維度災難
cardinality = train[cat_cols].nunique()
plt.figure(figsize=(8, 4))
plt.plot(cardinality.index, cardinality.values, marker='o', color='red')
for i, v in enumerate(cardinality.values):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.title('Cardinality of Categorical Columns')
plt.xlabel('Categorical Features')
plt.ylabel('Number of Unique Values')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 定義繪製類別分佈長條圖的輔助函式
def plot_category_distribution(df, col, title=None, cmap=plt.cm.Reds_r):
    counts = df[col].value_counts()
    plt.figure(figsize=(8, 5))
    colors = cmap(np.linspace(0, 1, len(counts)))
    plt.barh(counts.index, counts.values, color=colors, edgecolor='black')
    plt.gca().invert_yaxis() # 讓數量最多的排在最上方
    for i, v in enumerate(counts.values):
        plt.text(v, i, str(v), va='center', color='black')
    plt.title(title if title else f'{col} Distribution', fontsize=14)
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# 觀察輪胎配方與賽事的資料分佈
plot_category_distribution(train, 'Compound')
plot_category_distribution(train, 'Race', cmap=plt.cm.Blues_r)

# ── 2.2 Pre-Season Testing vs Actual Race 分布比較 ────────────────────────────
# 建立一個臨時標籤，區分「季前測試」與「正式比賽」
train['Race_type'] = np.where(
    train['Race'] == 'Pre-Season Testing', 'Pre-Season Testing', 'Actual Race'
)

# 觀察季前測試與正式比賽在各個數值特徵上的核密度估計 (KDE) 分佈差異
plot_cols = [
    'PitNextLap', 'PitStop', 'Stint', 'LapNumber', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change'
]
n = len(plot_cols)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
for i, col in enumerate(plot_cols):
    sns.kdeplot(data=train, x=col, hue='Race_type', fill=True, common_norm=True, ax=axes[i])
    axes[i].set_title(f"{col} Distribution")
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j]) # 刪除多餘的空白子圖
plt.tight_layout()
plt.show()

# 檢查不同賽事中使用了哪些輪胎配方
print("\n=== Race × Compound Crosstab ===")
print(pd.crosstab(train['Race'], train['Compound']))

# ── 2.3 數值欄位相關性熱圖 ────────────────────────────────────────────────────
# 計算數值特徵之間的 Pearson 相關係數，尋找多重共線性或與目標變數的高相關特徵
num_cols_target = [
    'PitNextLap', 'Stint', 'LapNumber', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change'
]
corr_matrix = train[num_cols_target].corr()
plt.figure(figsize=(12, 10))
plt.title('Feature Correlation Heatmap', fontsize=16)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.tight_layout()
plt.show()

# ── 2.4 數值欄位 KDE（依目標變數 PitNextLap 分色）────────────────────────────
# 觀察在「下一圈是否進站」的不同情況下，各數值特徵的分佈狀況（尋找有區分力的特徵）
features_num = train.select_dtypes(include="number").columns
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
axes = axes.flatten()
for ax, col in zip(axes, features_num):
    sns.kdeplot(data=train, x=col, hue="PitNextLap",
                fill=True, palette='bwr', common_norm=False, alpha=0.4, ax=ax)
    ax.set_title(col)
    ax.set_ylabel("Density")
for ax in axes[len(features_num):]:
    ax.set_visible(False)
fig.suptitle("KDE Distribution of Numerical Variables by PitNextLap", fontsize=20, y=1.02)
plt.tight_layout()
plt.show()

# ── 2.5 Compound × Avg Stint 熱圖 ────────────────────────────────────────────
# 觀察不同輪胎配方 (Compound) 平均能撐幾個 Stint (賽段)
ct = train.groupby('Compound').agg(avg_stint=('Stint', 'mean'))
plt.figure(figsize=(10, 6))
sns.heatmap(ct, annot=True, fmt=".4f", cmap="Blues_r")
plt.title("Compound vs Avg Stint")
plt.tight_layout()
plt.show()

# ── 2.6 數值欄位 KDE（PitStop 版）────────────────────────────────────────────
tmp_cols = [
    'PitStop', 'Stint', 'LapNumber', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change'
]
n = len(tmp_cols)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
for i, col in enumerate(tmp_cols):
    sns.kdeplot(data=train, x=col, hue='PitNextLap', fill=True, common_norm=True, ax=axes[i])
    axes[i].set_title(f"{col} by PitNextLap")
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# ── 2.7 類別欄位 vs PitNextLap 堆疊長條圖 ────────────────────────────────────
# 觀察不同類別 (如輪胎配方、賽事) 下，進站與不進站的比例差異
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
for i, col in enumerate(cat_cols[1:]): # 跳過第一個可能的 ID 類欄位
    ct = pd.crosstab(train[col], train['PitNextLap'], normalize='index')
    ct.plot(kind='bar', stacked=True, ax=axes[i])
    axes[i].set_title(f"{col} vs PitNextLap")
    axes[i].set_ylabel("Proportion")
    axes[i].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()

# ── 2.8 Hexbin 聯合分佈圖 ──────────────────────────────────────────────────────
# 處理點重疊嚴重的連續變數，觀察資料密集的熱區
def plot_hexbin(df, x_col, y_col, title=None, gridsize=20):
    plt.figure(figsize=(8, 5))
    hb = plt.hexbin(df[x_col], df[y_col], gridsize=gridsize, mincnt=1, cmap='viridis')
    plt.colorbar(hb).set_label('Count')
    plt.title(title if title else f'{x_col} vs {y_col}', fontsize=14)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# 觀察輪胎壽命與比賽圈數、圈數與賽程進度的聯合分佈
plot_hexbin(train, 'TyreLife', 'LapNumber')
plot_hexbin(train, 'LapNumber', 'RaceProgress')

# EDA 結束，清除為繪圖建立的暫時欄位
train = train.drop(columns=['Race_type'], errors='ignore')

# =============================================================================
# 3. Encoding — 嚴格遵循 fit on train, transform train + test 避免資料穿越
# =============================================================================
print("\n" + "=" * 60)
print("Encoding（fit on train only）")
print("=" * 60)

# ── 3.1 Compound → OrdinalEncoder ────────────────────────────────────────────
# 輪胎硬度有順序性 (SOFT 最軟 -> WET 最大/最特殊)，適合用 Ordinal Encoding
compound_order = [['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']]
oe_compound = OrdinalEncoder(
    categories=compound_order,
    handle_unknown='use_encoded_value',
    unknown_value=-1 # 若 test 出現 train 沒看過的輪胎，設為 -1
)
oe_compound.fit(train[['Compound']])                               # 僅在 train 擬合
train['Compound_encoded'] = oe_compound.transform(train[['Compound']])
test['Compound_encoded']  = oe_compound.transform(test[['Compound']])

print("\n[Compound → Compound_encoded]")
print(train[['Compound', 'Compound_encoded']].drop_duplicates().sort_values('Compound_encoded'))

# ── 3.2 Race → LabelEncoder ───────────────────────────────────────────────────
# 賽事名稱無大小順序之分，轉為數字 ID
le_race = LabelEncoder()
le_race.fit(train['Race'])                                         # 僅在 train 擬合
train['Race_encoded'] = le_race.transform(train['Race'])
# test 若有 train 沒見過的 Race，用 -1 補（確保測試集不會報錯）
test['Race_encoded'] = test['Race'].apply(
    lambda x: le_race.transform([x])[0] if x in le_race.classes_ else -1
)

print("\n[Race → Race_encoded]（前10筆）")
print(train[['Race', 'Race_encoded']].drop_duplicates().head(10))

# ── 3.3 Driver → LabelEncoder ─────────────────────────────────────────────────
# 車手名稱同理，轉為數字 ID
le_driver = LabelEncoder()
le_driver.fit(train['Driver'])                                     # 僅在 train 擬合
train['Driver_encoded'] = le_driver.transform(train['Driver'])
test['Driver_encoded'] = test['Driver'].apply(
    lambda x: le_driver.transform([x])[0] if x in le_driver.classes_ else -1
)

print(f"\n[Driver → Driver_encoded]  共 {train['Driver_encoded'].nunique()} 位車手")

# =============================================================================
# 4. Feature Engineering — 共用函式套用到 train / test (特徵工程)
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Pre-Season 旗標：區別是否為測試賽 (策略通常不同於正賽)
    df['Is_PreSeason'] = (df['Race'] == 'Pre-Season Testing').astype(int)

    # === 輪胎相關特徵 ===
    # 平均每個 stint 消耗的輪胎壽命 (+1 避免除以 0)
    df['TyreLife_per_Stint']      = df['TyreLife'] / (df['Stint'] + 1)
    # 輪胎壽命的非線性轉換 (衰退通常是非線性的)
    df['TyreLife_squared']        = df['TyreLife'] ** 2
    # 輪胎壽命與輪胎配方的交互作用 (硬胎跟軟胎的相同壽命意義不同)
    df['TyreLife_x_Compound']     = df['TyreLife'] * df['Compound_encoded']

    # === 圈速相關特徵 ===
    # 計算過去 3 圈的移動平均圈速 (依 Race + Stint 分組，絕對不能跨越進站前後計算)
    df['LapTime_rolling3']        = (
        df.groupby(['Race', 'Stint'])['LapTime (s)']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    # 當前圈速與近 3 圈平均的落差 (若為正值，代表單圈變慢，可能需進站)
    df['LapTime_vs_rolling3']     = df['LapTime (s)'] - df['LapTime_rolling3']

    # === 位置與賽程特徵 ===
    # 該賽段內的累積名次變化 (看這套輪胎有沒有幫助車手爬升)
    df['Position_Change_cumsum']  = (
        df.groupby(['Race', 'Stint'])['Position_Change']
        .transform('cumsum')
    )

    # === 賽事進度相關 ===
    # 距離上次進站過了幾圈
    df['Laps_Since_LastPit']      = (
        df.groupby(['Race', 'Stint'])['LapNumber']
        .transform(lambda x: x - x.min())
    )
    # 賽程進度的非線性轉換 (越接近比賽尾聲，進站機率變化可能加劇)
    df['RaceProgress_squared']    = df['RaceProgress'] ** 2

    # === 交互項 ===
    # 賽事進度與輪胎壽命的綜合影響
    df['TyreLife_x_RaceProgress'] = df['TyreLife'] * df['RaceProgress']

    return df

# 將特徵工程套用至 train 與 test
train = add_features(train)
test  = add_features(test)

new_cols = [
    'Is_PreSeason',
    'TyreLife_per_Stint', 'TyreLife_squared', 'TyreLife_x_Compound',
    'LapTime_rolling3', 'LapTime_vs_rolling3',
    'Position_Change_cumsum', 'Laps_Since_LastPit',
    'RaceProgress_squared', 'TyreLife_x_RaceProgress'
]
print("\n[新增欄位統計 — train]")
print(train[new_cols].describe().T[['mean', 'std', 'min', 'max']])
print("\n[新增欄位統計 — test]")
print(test[new_cols].describe().T[['mean', 'std', 'min', 'max']])

# ── 4.1 新特徵 × PitNextLap KDE 視覺化（train）────────────────────────────────
# 檢驗我們自己創造出來的新特徵，對目標變數 (PitNextLap) 是否有良好的鑑別度
n = len(new_cols)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
for i, col in enumerate(new_cols):
    sns.kdeplot(data=train, x=col, hue='PitNextLap', fill=True, common_norm=True, ax=axes[i])
    axes[i].set_title(col)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("New Features KDE by PitNextLap (train)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ── 4.2 最終相關性熱圖（train，含新特徵）─────────────────────────────────────
# 納入編碼後與特徵工程產生的新欄位，檢視最終的特徵關聯性
final_num_cols = [
    'PitNextLap', 'Stint', 'LapNumber', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change',
    'Compound_encoded', 'Race_encoded', 'Driver_encoded', 'Is_PreSeason',
    'TyreLife_per_Stint', 'TyreLife_squared', 'TyreLife_x_Compound',
    'LapTime_rolling3', 'LapTime_vs_rolling3',
    'Laps_Since_LastPit', 'RaceProgress_squared', 'TyreLife_x_RaceProgress'
]
corr2 = train[final_num_cols].corr()
plt.figure(figsize=(18, 16))
plt.title('Updated Correlation Heatmap (with Engineered Features)', fontsize=16)
sns.heatmap(corr2, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.tight_layout()
plt.show()

# ── 4.3 與目標變數相關性排行 ──────────────────────────────────────────────────
# 將各特徵與 PitNextLap 的相關係數抽出來，依據絕對值由強至弱排序
target_corr = corr2['PitNextLap'].drop('PitNextLap').sort_values(key=abs, ascending=False)
print("\n[與 PitNextLap 相關係數（由強至弱）]")
print(target_corr.to_string())

plt.figure(figsize=(10, 7))
# 畫出長條圖，正相關用紅色(tomato)，負相關用藍色(steelblue)
colors = ['tomato' if v > 0 else 'steelblue' for v in target_corr.values]
plt.barh(target_corr.index[::-1], target_corr.values[::-1], color=colors[::-1])
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Feature Correlation with PitNextLap', fontsize=14)
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.show()

# =============================================================================
# 5. 輸出 (存檔以供模型訓練使用)
# =============================================================================
print("\n=== Train 最終欄位 ===")
print(train.columns.to_list())
print(f"Shape：{train.shape}")

print("\n=== Test 最終欄位 ===")
print(test.columns.to_list())
print(f"Shape：{test.shape}")

# 將處理乾淨且富含新特徵的資料集存檔
train.to_csv('dataset/train_engineered.csv', index=False)
test.to_csv('dataset/test_engineered.csv',  index=False)
=======
# =============================================================================
# F1 Pit Stop Prediction — EDA + Feature Engineering (資料探索與特徵工程)
# =============================================================================

# ── 0. Imports ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# ── 1. Load Data ──────────────────────────────────────────────────────────────
# 載入訓練集與測試集
train = pd.read_csv('dataset/train.csv')
test  = pd.read_csv('dataset/test.csv')

# 基礎資料探索：檢視資料型態、統計數值與缺失值狀況
train.info()
print(train.describe())
print("Train Nulls:\n", train.isnull().sum())
print("Test Nulls:\n", test.isnull().sum())

# 分離欄位類型（以 train 為基準）
# 自動將欄位分為類別型 (Categorical) 與數值型 (Numerical)，方便後續批次處理
cat_cols = train.select_dtypes(exclude=['number']).columns.to_list()
num_cols = train.select_dtypes(include=['number']).columns.to_list()
print(f"\nCategorical Columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical Columns   ({len(num_cols)}): {num_cols}")


# ── 2.1 類別欄位基數 (Cardinality) ──────────────────────────────────────────────
# 觀察類別欄位中有多少種不同的獨立值 (Unique values)，評估是否會有維度災難
cardinality = train[cat_cols].nunique()
plt.figure(figsize=(8, 4))
plt.plot(cardinality.index, cardinality.values, marker='o', color='red')
for i, v in enumerate(cardinality.values):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.title('Cardinality of Categorical Columns')
plt.xlabel('Categorical Features')
plt.ylabel('Number of Unique Values')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 定義繪製類別分佈長條圖的輔助函式
def plot_category_distribution(df, col, title=None, cmap=plt.cm.Reds_r):
    counts = df[col].value_counts()
    plt.figure(figsize=(8, 5))
    colors = cmap(np.linspace(0, 1, len(counts)))
    plt.barh(counts.index, counts.values, color=colors, edgecolor='black')
    plt.gca().invert_yaxis() # 讓數量最多的排在最上方
    for i, v in enumerate(counts.values):
        plt.text(v, i, str(v), va='center', color='black')
    plt.title(title if title else f'{col} Distribution', fontsize=14)
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# 觀察輪胎配方與賽事的資料分佈
plot_category_distribution(train, 'Compound')
plot_category_distribution(train, 'Race', cmap=plt.cm.Blues_r)

# ── 2.2 Pre-Season Testing vs Actual Race 分布比較 ────────────────────────────
# 建立一個臨時標籤，區分「季前測試」與「正式比賽」
train['Race_type'] = np.where(
    train['Race'] == 'Pre-Season Testing', 'Pre-Season Testing', 'Actual Race'
)

# 觀察季前測試與正式比賽在各個數值特徵上的核密度估計 (KDE) 分佈差異
plot_cols = [
    'PitNextLap', 'PitStop', 'Stint', 'LapNumber', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change'
]
n = len(plot_cols)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
for i, col in enumerate(plot_cols):
    sns.kdeplot(data=train, x=col, hue='Race_type', fill=True, common_norm=True, ax=axes[i])
    axes[i].set_title(f"{col} Distribution")
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j]) # 刪除多餘的空白子圖
plt.tight_layout()
plt.show()

# 檢查不同賽事中使用了哪些輪胎配方
print("\n=== Race × Compound Crosstab ===")
print(pd.crosstab(train['Race'], train['Compound']))

# ── 2.3 數值欄位相關性熱圖 ────────────────────────────────────────────────────
# 計算數值特徵之間的 Pearson 相關係數，尋找多重共線性或與目標變數的高相關特徵
num_cols_target = [
    'PitNextLap', 'Stint', 'LapNumber', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change'
]
corr_matrix = train[num_cols_target].corr()
plt.figure(figsize=(12, 10))
plt.title('Feature Correlation Heatmap', fontsize=16)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.tight_layout()
plt.show()

# ── 2.4 數值欄位 KDE（依目標變數 PitNextLap 分色）────────────────────────────
# 觀察在「下一圈是否進站」的不同情況下，各數值特徵的分佈狀況（尋找有區分力的特徵）
features_num = train.select_dtypes(include="number").columns
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
axes = axes.flatten()
for ax, col in zip(axes, features_num):
    sns.kdeplot(data=train, x=col, hue="PitNextLap",
                fill=True, palette='bwr', common_norm=False, alpha=0.4, ax=ax)
    ax.set_title(col)
    ax.set_ylabel("Density")
for ax in axes[len(features_num):]:
    ax.set_visible(False)
fig.suptitle("KDE Distribution of Numerical Variables by PitNextLap", fontsize=20, y=1.02)
plt.tight_layout()
plt.show()

# ── 2.5 Compound × Avg Stint 熱圖 ────────────────────────────────────────────
# 觀察不同輪胎配方 (Compound) 平均能撐幾個 Stint (賽段)
ct = train.groupby('Compound').agg(avg_stint=('Stint', 'mean'))
plt.figure(figsize=(10, 6))
sns.heatmap(ct, annot=True, fmt=".4f", cmap="Blues_r")
plt.title("Compound vs Avg Stint")
plt.tight_layout()
plt.show()

# ── 2.6 數值欄位 KDE（PitStop 版）────────────────────────────────────────────
tmp_cols = [
    'PitStop', 'Stint', 'LapNumber', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change'
]
n = len(tmp_cols)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
for i, col in enumerate(tmp_cols):
    sns.kdeplot(data=train, x=col, hue='PitNextLap', fill=True, common_norm=True, ax=axes[i])
    axes[i].set_title(f"{col} by PitNextLap")
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.show()

# ── 2.7 類別欄位 vs PitNextLap 堆疊長條圖 ────────────────────────────────────
# 觀察不同類別 (如輪胎配方、賽事) 下，進站與不進站的比例差異
fig, axes = plt.subplots(1, 2, figsize=(18, 5))
for i, col in enumerate(cat_cols[1:]): # 跳過第一個可能的 ID 類欄位
    ct = pd.crosstab(train[col], train['PitNextLap'], normalize='index')
    ct.plot(kind='bar', stacked=True, ax=axes[i])
    axes[i].set_title(f"{col} vs PitNextLap")
    axes[i].set_ylabel("Proportion")
    axes[i].tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()

# ── 2.8 Hexbin 聯合分佈圖 ──────────────────────────────────────────────────────
# 處理點重疊嚴重的連續變數，觀察資料密集的熱區
def plot_hexbin(df, x_col, y_col, title=None, gridsize=20):
    plt.figure(figsize=(8, 5))
    hb = plt.hexbin(df[x_col], df[y_col], gridsize=gridsize, mincnt=1, cmap='viridis')
    plt.colorbar(hb).set_label('Count')
    plt.title(title if title else f'{x_col} vs {y_col}', fontsize=14)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# 觀察輪胎壽命與比賽圈數、圈數與賽程進度的聯合分佈
plot_hexbin(train, 'TyreLife', 'LapNumber')
plot_hexbin(train, 'LapNumber', 'RaceProgress')

# EDA 結束，清除為繪圖建立的暫時欄位
train = train.drop(columns=['Race_type'], errors='ignore')

# =============================================================================
# 3. Encoding — 嚴格遵循 fit on train, transform train + test 避免資料穿越
# =============================================================================
print("\n" + "=" * 60)
print("Encoding（fit on train only）")
print("=" * 60)

# ── 3.1 Compound → OrdinalEncoder ────────────────────────────────────────────
# 輪胎硬度有順序性 (SOFT 最軟 -> WET 最大/最特殊)，適合用 Ordinal Encoding
compound_order = [['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']]
oe_compound = OrdinalEncoder(
    categories=compound_order,
    handle_unknown='use_encoded_value',
    unknown_value=-1 # 若 test 出現 train 沒看過的輪胎，設為 -1
)
oe_compound.fit(train[['Compound']])                               # 僅在 train 擬合
train['Compound_encoded'] = oe_compound.transform(train[['Compound']])
test['Compound_encoded']  = oe_compound.transform(test[['Compound']])

print("\n[Compound → Compound_encoded]")
print(train[['Compound', 'Compound_encoded']].drop_duplicates().sort_values('Compound_encoded'))

# ── 3.2 Race → LabelEncoder ───────────────────────────────────────────────────
# 賽事名稱無大小順序之分，轉為數字 ID
le_race = LabelEncoder()
le_race.fit(train['Race'])                                         # 僅在 train 擬合
train['Race_encoded'] = le_race.transform(train['Race'])
# test 若有 train 沒見過的 Race，用 -1 補（確保測試集不會報錯）
test['Race_encoded'] = test['Race'].apply(
    lambda x: le_race.transform([x])[0] if x in le_race.classes_ else -1
)

print("\n[Race → Race_encoded]（前10筆）")
print(train[['Race', 'Race_encoded']].drop_duplicates().head(10))

# ── 3.3 Driver → LabelEncoder ─────────────────────────────────────────────────
# 車手名稱同理，轉為數字 ID
le_driver = LabelEncoder()
le_driver.fit(train['Driver'])                                     # 僅在 train 擬合
train['Driver_encoded'] = le_driver.transform(train['Driver'])
test['Driver_encoded'] = test['Driver'].apply(
    lambda x: le_driver.transform([x])[0] if x in le_driver.classes_ else -1
)

print(f"\n[Driver → Driver_encoded]  共 {train['Driver_encoded'].nunique()} 位車手")

# =============================================================================
# 4. Feature Engineering — 共用函式套用到 train / test (特徵工程)
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Pre-Season 旗標：區別是否為測試賽 (策略通常不同於正賽)
    df['Is_PreSeason'] = (df['Race'] == 'Pre-Season Testing').astype(int)

    # === 輪胎相關特徵 ===
    # 平均每個 stint 消耗的輪胎壽命 (+1 避免除以 0)
    df['TyreLife_per_Stint']      = df['TyreLife'] / (df['Stint'] + 1)
    # 輪胎壽命的非線性轉換 (衰退通常是非線性的)
    df['TyreLife_squared']        = df['TyreLife'] ** 2
    # 輪胎壽命與輪胎配方的交互作用 (硬胎跟軟胎的相同壽命意義不同)
    df['TyreLife_x_Compound']     = df['TyreLife'] * df['Compound_encoded']

    # === 圈速相關特徵 ===
    # 計算過去 3 圈的移動平均圈速 (依 Race + Stint 分組，絕對不能跨越進站前後計算)
    df['LapTime_rolling3']        = (
        df.groupby(['Race', 'Stint'])['LapTime (s)']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    # 當前圈速與近 3 圈平均的落差 (若為正值，代表單圈變慢，可能需進站)
    df['LapTime_vs_rolling3']     = df['LapTime (s)'] - df['LapTime_rolling3']

    # === 位置與賽程特徵 ===
    # 該賽段內的累積名次變化 (看這套輪胎有沒有幫助車手爬升)
    df['Position_Change_cumsum']  = (
        df.groupby(['Race', 'Stint'])['Position_Change']
        .transform('cumsum')
    )

    # === 賽事進度相關 ===
    # 距離上次進站過了幾圈
    df['Laps_Since_LastPit']      = (
        df.groupby(['Race', 'Stint'])['LapNumber']
        .transform(lambda x: x - x.min())
    )
    # 賽程進度的非線性轉換 (越接近比賽尾聲，進站機率變化可能加劇)
    df['RaceProgress_squared']    = df['RaceProgress'] ** 2

    # === 交互項 ===
    # 賽事進度與輪胎壽命的綜合影響
    df['TyreLife_x_RaceProgress'] = df['TyreLife'] * df['RaceProgress']

    return df

# 將特徵工程套用至 train 與 test
train = add_features(train)
test  = add_features(test)

new_cols = [
    'Is_PreSeason',
    'TyreLife_per_Stint', 'TyreLife_squared', 'TyreLife_x_Compound',
    'LapTime_rolling3', 'LapTime_vs_rolling3',
    'Position_Change_cumsum', 'Laps_Since_LastPit',
    'RaceProgress_squared', 'TyreLife_x_RaceProgress'
]
print("\n[新增欄位統計 — train]")
print(train[new_cols].describe().T[['mean', 'std', 'min', 'max']])
print("\n[新增欄位統計 — test]")
print(test[new_cols].describe().T[['mean', 'std', 'min', 'max']])

# ── 4.1 新特徵 × PitNextLap KDE 視覺化（train）────────────────────────────────
# 檢驗我們自己創造出來的新特徵，對目標變數 (PitNextLap) 是否有良好的鑑別度
n = len(new_cols)
cols = 3
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()
for i, col in enumerate(new_cols):
    sns.kdeplot(data=train, x=col, hue='PitNextLap', fill=True, common_norm=True, ax=axes[i])
    axes[i].set_title(col)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
fig.suptitle("New Features KDE by PitNextLap (train)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ── 4.2 最終相關性熱圖（train，含新特徵）─────────────────────────────────────
# 納入編碼後與特徵工程產生的新欄位，檢視最終的特徵關聯性
final_num_cols = [
    'PitNextLap', 'Stint', 'LapNumber', 'TyreLife', 'Position',
    'LapTime (s)', 'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position_Change',
    'Compound_encoded', 'Race_encoded', 'Driver_encoded', 'Is_PreSeason',
    'TyreLife_per_Stint', 'TyreLife_squared', 'TyreLife_x_Compound',
    'LapTime_rolling3', 'LapTime_vs_rolling3',
    'Laps_Since_LastPit', 'RaceProgress_squared', 'TyreLife_x_RaceProgress'
]
corr2 = train[final_num_cols].corr()
plt.figure(figsize=(18, 16))
plt.title('Updated Correlation Heatmap (with Engineered Features)', fontsize=16)
sns.heatmap(corr2, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.tight_layout()
plt.show()

# ── 4.3 與目標變數相關性排行 ──────────────────────────────────────────────────
# 將各特徵與 PitNextLap 的相關係數抽出來，依據絕對值由強至弱排序
target_corr = corr2['PitNextLap'].drop('PitNextLap').sort_values(key=abs, ascending=False)
print("\n[與 PitNextLap 相關係數（由強至弱）]")
print(target_corr.to_string())

plt.figure(figsize=(10, 7))
# 畫出長條圖，正相關用紅色(tomato)，負相關用藍色(steelblue)
colors = ['tomato' if v > 0 else 'steelblue' for v in target_corr.values]
plt.barh(target_corr.index[::-1], target_corr.values[::-1], color=colors[::-1])
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Feature Correlation with PitNextLap', fontsize=14)
plt.xlabel('Pearson Correlation')
plt.tight_layout()
plt.show()

# =============================================================================
# 5. 輸出 (存檔以供模型訓練使用)
# =============================================================================
print("\n=== Train 最終欄位 ===")
print(train.columns.to_list())
print(f"Shape：{train.shape}")

print("\n=== Test 最終欄位 ===")
print(test.columns.to_list())
print(f"Shape：{test.shape}")

# 將處理乾淨且富含新特徵的資料集存檔
train.to_csv('dataset/train_engineered.csv', index=False)
test.to_csv('dataset/test_engineered.csv',  index=False)
>>>>>>> a47810e95c4fe9d47c915bcc6ad4fd2a55af2fce
print("\n✅ 已儲存：dataset/train_engineered.csv & dataset/test_engineered.csv")