import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Bắt đầu xử lý dữ liệu chấm điểm rủi ro tín dụng...")

# 1. Đọc dữ liệu
loan = pd.read_csv('data/berka/loan.csv', sep=';')
account = pd.read_csv('data/berka/account.csv', sep=';')
trans = pd.read_csv('data/berka/trans.csv', sep=';', low_memory=False)
district = pd.read_csv('data/berka/district.csv', sep=';')

# Chuyển đổi ngày
loan['loan_date'] = pd.to_datetime('19' + loan['date'].astype(str).str.zfill(6), format='%Y%m%d')
account['account_date'] = pd.to_datetime('19' + account['date'].astype(str).str.zfill(6), format='%Y%m%d')
trans['trans_date'] = pd.to_datetime('19' + trans['date'].astype(str).str.zfill(6), format='%Y%m%d')

# Ghép thông tin huyện vào account
account = account.merge(district[['A1', 'A11', 'A12', 'A13']], left_on='district_id', right_on='A1', how='left')
loan = loan.merge(account[['account_id', 'account_date', 'district_id', 'A11', 'A13']], on='account_id', how='left')
loan['DaysToLoan'] = (loan['loan_date'] - loan['account_date']).dt.days

# Lọc giao dịch của các tài khoản có vay
loan_accounts = set(loan['account_id'].unique())
trans_loan = trans[trans['account_id'].isin(loan_accounts)].copy()

# 2. FEATURE ENGINEERING TRƯỚC NGÀY VAY (Chống rò rỉ dữ liệu)
features = []
for _, row in loan.iterrows():
    acc_id = row['account_id']
    l_date = row['loan_date']
    
    # Chỉ lấy giao dịch TRƯỚC ngày vay
    sub_trans = trans_loan[(trans_loan['account_id'] == acc_id) & (trans_loan['trans_date'] < l_date)]
    d90_start = l_date - pd.Timedelta(days=90)
    sub_90d = sub_trans[sub_trans['trans_date'] >= d90_start]
    
    if len(sub_90d) > 0:
        avg_bal = sub_90d['balance'].mean()
        min_bal = sub_90d['balance'].min()
        neg_count = (sub_90d['balance'] < 0).sum()
        tx_count = len(sub_90d)
        credits_90d = sub_90d[sub_90d['type'] == 'PRIJEM']['amount'].sum()
        monthly_inc = credits_90d / 3.0
    else:
        avg_bal = sub_trans['balance'].mean() if len(sub_trans) > 0 else 0
        min_bal = sub_trans['balance'].min() if len(sub_trans) > 0 else 0
        neg_count = (sub_trans['balance'] < 0).sum() if len(sub_trans) > 0 else 0
        tx_count = len(sub_trans)
        credits_all = sub_trans[sub_trans['type'] == 'PRIJEM']['amount'].sum() if len(sub_trans) > 0 else 0
        monthly_inc = credits_all / max(1, (len(sub_trans) / 30.0))
        
    dsr = row['payments'] / monthly_inc if monthly_inc > 0 else 1.0
    dsr = min(dsr, 2.5) # giới hạn outlier
    
    salary = float(str(row['A11']).replace('?', '0')) if pd.notnull(row['A11']) else 8000.0
    unemp = float(str(row['A13']).replace('?', '0')) if pd.notnull(row['A13']) else 3.0
    
    features.append({
        'loan_id': row['loan_id'],
        'account_id': acc_id,
        'DaysToLoan': row['DaysToLoan'],
        'LoanAmount': row['amount'],
        'LoanDuration': row['duration'],
        'MonthlyPayment': row['payments'],
        'AvgBalance_90d': avg_bal,
        'MinBalance_90d': min_bal,
        'NegBalanceCount_90d': neg_count,
        'TxCount_90d': tx_count,
        'MonthlyIncomeEst': monthly_inc,
        'DSR': dsr,
        'District_Salary': salary,
        'District_Unemp': unemp,
        'Status': row['status'],
        'Default': 1 if row['status'] in ['B', 'D'] else 0
    })

df = pd.DataFrame(features)
print(f"Đã tạo {len(df)} dòng dữ liệu đặc trưng.")
print(f"Số khoản vỡ nợ (B, D): {df['Default'].sum()} ({df['Default'].mean()*100:.2f}%)")

# 3. CHUẨN BỊ X VÀ Y
feature_cols = [
    'DaysToLoan', 'LoanAmount', 'LoanDuration', 'MonthlyPayment',
    'AvgBalance_90d', 'MinBalance_90d', 'NegBalanceCount_90d',
    'TxCount_90d', 'MonthlyIncomeEst', 'DSR',
    'District_Salary', 'District_Unemp'
]

X_raw = df[feature_cols].values
y = df['Default'].values

# Chuẩn hóa Z-score
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0) + 1e-8
X_norm = (X_raw - X_mean) / X_std

# Hàm tính AUC-ROC chính xác bằng Wilcoxon rank-sum
def compute_auc(y_true, y_scores):
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # So sánh từng cặp pos và neg
    return np.mean([1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg])

# 4. MÔ HÌNH 1: LOGISTIC REGRESSION (với Class Weight Balanced)
class LogisticRegressionPure:
    def __init__(self, lr=0.05, epochs=300, l2=0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None
        self.b = 0.0
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Trọng số cân bằng lớp
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        weight_pos = n_neg / n_pos
        weights = np.where(y == 1, weight_pos, 1.0)
        
        for _ in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))
            
            error = (p - y) * weights
            dw = (np.dot(X.T, error) / n_samples) + self.l2 * self.w
            db = np.sum(error) / n_samples
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15, 15)))

# 5. MÔ HÌNH 2: DECISION TREE CLASSIFIER (với Gini & Class Weight)
class DecisionTreePure:
    def __init__(self, max_depth=4, min_samples_split=15):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        # Trọng số
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        w_pos = n_neg / max(1, n_pos)
        weights = np.where(y == 1, w_pos, 1.0)
        self.tree = self._build_tree(X, y, weights, depth=0)

    def _gini(self, y, weights):
        total_w = np.sum(weights)
        if total_w == 0:
            return 0
        p1 = np.sum(weights[y == 1]) / total_w
        return 1.0 - (p1**2 + (1.0 - p1)**2)

    def _build_tree(self, X, y, weights, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            p_pos = np.sum(weights[y == 1]) / np.sum(weights) if np.sum(weights) > 0 else 0
            return {'leaf': True, 'prob': p_pos}

        best_gini = 999.0
        best_split = None
        base_gini = self._gini(y, weights)

        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) > 20:
                thresholds = np.percentile(vals, np.linspace(10, 90, 9))
            else:
                thresholds = vals
                
            for thresh in thresholds:
                left_idx = X[:, feat] <= thresh
                right_idx = ~left_idx
                if np.sum(left_idx) < 5 or np.sum(right_idx) < 5:
                    continue
                w_left, w_right = weights[left_idx], weights[right_idx]
                gini = (np.sum(w_left) * self._gini(y[left_idx], w_left) + np.sum(w_right) * self._gini(y[right_idx], w_right)) / np.sum(weights)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feat, thresh, left_idx, right_idx)

        if best_split is None or best_gini >= base_gini:
            p_pos = np.sum(weights[y == 1]) / np.sum(weights) if np.sum(weights) > 0 else 0
            return {'leaf': True, 'prob': p_pos}

        feat, thresh, left_idx, right_idx = best_split
        return {
            'leaf': False,
            'feature': feat,
            'threshold': thresh,
            'left': self._build_tree(X[left_idx], y[left_idx], weights[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], y[right_idx], weights[right_idx], depth + 1)
        }

    def _predict_one(self, node, x):
        if node['leaf']:
            return node['prob']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(node['left'], x)
        else:
            return self._predict_one(node['right'], x)

    def predict_proba(self, X):
        return np.array([self._predict_one(self.tree, row) for row in X])

# 6. ĐÁNH GIÁ 5-FOLD STRATIFIED CROSS-VALIDATION
np.random.seed(42)
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
np.random.shuffle(pos_idx)
np.random.shuffle(neg_idx)

k_folds = 5
folds = []
for k in range(k_folds):
    test_pos = pos_idx[k * len(pos_idx)//k_folds : (k+1) * len(pos_idx)//k_folds]
    test_neg = neg_idx[k * len(neg_idx)//k_folds : (k+1) * len(neg_idx)//k_folds]
    test_idx = np.concatenate([test_pos, test_neg])
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    folds.append((train_idx, test_idx))

lr_aucs, dt_aucs = [], []
lr_recalls, dt_recalls = [], []

for train_idx, test_idx in folds:
    X_train, y_train = X_norm[train_idx], y[train_idx]
    X_test, y_test = X_norm[test_idx], y[test_idx]
    
    # Train Logistic Regression
    lr = LogisticRegressionPure(lr=0.08, epochs=350, l2=0.02)
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)
    lr_aucs.append(compute_auc(y_test, p_lr))
    lr_recalls.append(np.sum((p_lr >= 0.5) & (y_test == 1)) / np.sum(y_test == 1))
    
    # Train Decision Tree
    dt = DecisionTreePure(max_depth=4, min_samples_split=15)
    dt.fit(X_train, y_train)
    p_dt = dt.predict_proba(X_test)
    dt_aucs.append(compute_auc(y_test, p_dt))
    dt_recalls.append(np.sum((p_dt >= 0.5) & (y_test == 1)) / np.sum(y_test == 1))

print("\n" + "="*50)
print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (5-Fold Stratified Cross-Validation):")
print(f"1. Logistic Regression: AUC = {np.mean(lr_aucs):.4f} (±{np.std(lr_aucs):.4f}) | Recall = {np.mean(lr_recalls)*100:.1f}%")
print(f"2. Decision Tree      : AUC = {np.mean(dt_aucs):.4f} (±{np.std(dt_aucs):.4f}) | Recall = {np.mean(dt_recalls)*100:.1f}%")
print("="*50)

# 7. HUẤN LUYỆN TOÀN BỘ VÀ XUẤT ĐIỂM PD (Dùng Logistic Regression làm mô hình chốt vì AUC cao & ổn định)
final_model = LogisticRegressionPure(lr=0.08, epochs=400, l2=0.02)
final_model.fit(X_norm, y)
pd_scores = final_model.predict_proba(X_norm)

# Phân loại rủi ro (Risk Tier)
# - Thấp (Low Risk): PD < 0.25
# - Trung bình (Medium Risk): 0.25 <= PD < 0.55
# - Cao (High Risk): PD >= 0.55
risk_tiers = []
risk_order = []
for p in pd_scores:
    if p < 0.25:
        risk_tiers.append('Rủi ro thấp')
        risk_order.append(1)
    elif p < 0.55:
        risk_tiers.append('Rủi ro trung bình')
        risk_order.append(2)
    else:
        risk_tiers.append('Rủi ro cao')
        risk_order.append(3)

df['PD_Score'] = np.round(pd_scores, 4)
df['Risk_Group'] = risk_tiers
df['Risk_Order'] = risk_order
df['Predicted_Default'] = (pd_scores >= 0.5).astype(int)

# Ghi file kết quả cho Power BI
os.makedirs('data/berka', exist_ok=True)
output_cols = ['loan_id', 'account_id', 'LoanAmount', 'LoanDuration', 'MonthlyPayment',
               'DaysToLoan', 'AvgBalance_90d', 'MinBalance_90d', 'NegBalanceCount_90d',
               'DSR', 'Status', 'Default', 'PD_Score', 'Risk_Group', 'Risk_Order', 'Predicted_Default']

out_csv = 'data/berka/ml_loan_credit_scoring.csv'
df[output_cols].to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\nĐã xuất file kết quả: {out_csv} ({len(df)} dòng).")

# Tóm tắt phân loại rủi ro
print("\nPhân bổ rủi ro thực tế:")
summary = df.groupby('Risk_Group').agg(
    So_khoan_vay=('loan_id', 'count'),
    So_khoan_vo_no=('Default', 'sum'),
    Ty_le_vo_no_thuc_te=('Default', 'mean'),
    PD_Trung_Binh=('PD_Score', 'mean')
).reindex(['Rủi ro thấp', 'Rủi ro trung bình', 'Rủi ro cao'])
summary['Ty_le_vo_no_thuc_te'] = (summary['Ty_le_vo_no_thuc_te'] * 100).round(2).astype(str) + '%'
summary['PD_Trung_Binh'] = (summary['PD_Trung_Binh'] * 100).round(2).astype(str) + '%'
print(summary)

# 8. VẼ BIỂU ĐỒ ĐÁNH GIÁ (ROC Curve & Feature Importance)
os.makedirs('ml', exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

# Đường ROC (sắp xếp theo threshold)
fpr_list, tpr_list = [0.0], [0.0]
sorted_indices = np.argsort(pd_scores)[::-1]
y_sorted = y[sorted_indices]
tp, fp = 0, 0
total_p = np.sum(y == 1)
total_n = np.sum(y == 0)

for val in y_sorted:
    if val == 1:
        tp += 1
    else:
        fp += 1
    fpr_list.append(fp / total_n)
    tpr_list.append(tp / total_p)

ax[0].plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {np.mean(lr_aucs):.3f})')
ax[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.05])
ax[0].set_xlabel('False Positive Rate (FPR)')
ax[0].set_ylabel('True Positive Rate (Recall / TPR)')
ax[0].set_title('Đường cong ROC dự báo vỡ nợ')
ax[0].legend(loc="lower right")
ax[0].grid(True, alpha=0.3)

# Độ quan trọng đặc trưng (Hệ số Logistic Regression)
weights = np.abs(final_model.w)
sorted_feat_idx = np.argsort(weights)
ax[1].barh(np.array(feature_cols)[sorted_feat_idx], weights[sorted_feat_idx], color='steelblue')
ax[1].set_xlabel('Trọng số ảnh hưởng (|Beta Coefficient|)')
ax[1].set_title('Mức độ quan trọng của các biến dự báo')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = 'ml/credit_scoring_roc_importance.png'
plt.savefig(fig_path, dpi=150)
print(f"Đã lưu biểu đồ đánh giá: {fig_path}")
