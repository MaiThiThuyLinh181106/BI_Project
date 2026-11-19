import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def read_berka_csv(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        header = f.readline()
    sep = ';' if header.count(';') > header.count(',') else ','
    return pd.read_csv(path, sep=sep, low_memory=False)

def parse_berka_date(series):
    s = series.astype(str).str.strip()
    if s.iloc[0].isdigit() and len(s.iloc[0]) == 6:
        return pd.to_datetime('19' + s, format='%Y%m%d')
    return pd.to_datetime(s)

# đọc data
trans = read_berka_csv('data/berka/trans.csv')
account = read_berka_csv('data/berka/account.csv')
print(f"Đã đọc trans: {len(trans):,} dòng | account: {len(account):,} dòng")

trans['trans_date'] = parse_berka_date(trans['date'])
trans['k_symbol'] = trans['k_symbol'].fillna('').astype(str).str.strip()
snapshot_date = trans['trans_date'].max()
print(f"Khoảng thời gian giao dịch: {trans['trans_date'].min().date()} -> {snapshot_date.date()}")

# tính RFM năm 1998 (lọc bỏ giao dịch lãi/phí do ngân hàng tự ghi)
bank_auto = {'UROK', 'SLUZBY', 'SANKC. UROK'}
trans_98 = trans[trans['trans_date'] >= '1998-01-01']
cust_trans_98 = trans_98[~trans_98['k_symbol'].isin(bank_auto)]

rfm = cust_trans_98.groupby('account_id').agg(
    Recency=('trans_date', lambda x: (snapshot_date - x.max()).days),
    Frequency=('trans_id', 'count'),
    Monetary=('amount', 'sum')
).reset_index()

# tính số dư bình quân năm 1998
bal_98 = trans_98.groupby('account_id')['balance'].mean().reset_index()
bal_98.rename(columns={'balance': 'AvgBalance'}, inplace=True)

# ghép 4.500 tài khoản
df = account[['account_id']].merge(rfm, on='account_id', how='left').merge(bal_98, on='account_id', how='left')
df['Recency'] = df['Recency'].fillna(365)
df[['Frequency', 'Monetary', 'AvgBalance']] = df[['Frequency', 'Monetary', 'AvgBalance']].fillna(0)

# tiền xử lý và chuẩn hóa
X = df[['Recency', 'Frequency', 'Monetary', 'AvgBalance']].copy()
X['Recency'] = np.log1p(X['Recency'].clip(lower=0))
X['Frequency'] = np.log1p(X['Frequency'].clip(lower=0))
X['Monetary'] = np.log1p(X['Monetary'].clip(lower=0))
X_scaled = StandardScaler().fit_transform(X)

# chạy K-means với k = 2..8 để đánh giá
inertias, sil_scores = [], []
K_range = range(2, 9)
print("\nKết quả đánh giá số cụm k:")
print(f"{'k':>2} | {'Inertia (Elbow)':>16} | {'Silhouette':>10}")
print("-" * 35)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil = silhouette_score(X_scaled, labels)
    inertias.append(inertia)
    sil_scores.append(sil)
    print(f"{k:>2} | {inertia:>16.1f} | {sil:>10.4f}")

# lưu ảnh biểu đồ
os.makedirs('ml', exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(K_range, inertias, marker='o')
ax[0].set_title('Elbow Method (Inertia)')
ax[0].set_xlabel('Số cụm (k)')
ax[0].set_ylabel('Inertia')
ax[0].grid(True, alpha=0.3)

ax[1].plot(K_range, sil_scores, marker='o', color='green')
ax[1].set_title('Silhouette Score')
ax[1].set_xlabel('Số cụm (k)')
ax[1].set_ylabel('Silhouette Score')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml/elbow_silhouette.png', dpi=150)
print("\nĐã lưu biểu đồ tại: ml/elbow_silhouette.png")
# chạy mô hình chốt với k = 4 và xem hồ sơ từng cụm
best_k = 4
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster_ID'] = km_final.fit_predict(X_scaled)

profile = df.groupby('Cluster_ID').agg(
    So_TK=('account_id', 'count'),
    Recency_TB=('Recency', 'mean'),
    Frequency_TB=('Frequency', 'mean'),
    Monetary_TB=('Monetary', 'mean'),
    AvgBalance_TB=('AvgBalance', 'mean')
).round(1)

print("\nHồ sơ 4 cụm (k = 4):")
print(profile)
# đặt tên cụm, sắp xếp thứ tự, xuất csv
name_map = {
    3: 'Khách giá trị cao',
    1: 'Giao dịch tích cực',
    0: 'Khách phổ thông',
    2: 'Ít hoạt động'
}
order_map = {
    3: 1,
    1: 2,
    0: 3,
    2: 4
}

df['Segment_Name'] = df['Cluster_ID'].map(name_map)
df['Segment_Order'] = df['Cluster_ID'].map(order_map)

# làm tròn các chỉ số cho gọn trên powerbi
df['Monetary'] = df['Monetary'].round(2)
df['AvgBalance'] = df['AvgBalance'].round(2)

output_path = 'data/berka/ml_customer_segments.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nĐã xuất file kết quả tại: {output_path} ({len(df):,} dòng)")