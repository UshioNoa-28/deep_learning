import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

# --- 1. 数据加载 ---
train = pd.read_csv('Data/kaggle_house_pred_train.csv')
test = pd.read_csv('Data/kaggle_house_pred_test.csv')

# --- 2. 异常值处理 ---
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# --- 3. 标签转换 ---
y_train = np.log1p(train.SalePrice.values)
train_id = train.Id
test_id = test.Id
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# --- 4. 特征工程 ---
ntrain = train.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

# A. 精细化填充缺失值
# 类别型：缺失通常代表“没有”
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

# 数值型：缺失通常代表 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1',
            'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

# B. 核心安全垫：处理剩下的遗漏空值 (重点修复)
# 数值列用中位数填充
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].median())

# 类别列用众数填充
object_cols = all_data.select_dtypes(include=['object']).columns
for col in object_cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# C. 类型转换
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

# D. 偏度校正
numeric_data = all_data.select_dtypes(include=[np.number])
skewed_feats = numeric_data.apply(lambda x: x.skew()).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.75]
for feat in high_skew.index:
    all_data[feat] = np.log1p(all_data[feat])

# E. 独热编码 (One-Hot Encoding)
all_data = pd.get_dummies(all_data)
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

# --- 5. 模型构建 ---
# Lasso 增加容错控制
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

model_xgb = xgb.XGBRegressor(colsample_bytree=0.46, gamma=0.046,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.78, n_estimators=2200,
                             reg_alpha=0.46, reg_lambda=0.85,
                             subsample=0.52, random_state=7, nthread=-1)

# --- 6. 训练与预测 ---
print("Training Lasso (Strict Engine)...")
lasso.fit(X_train, y_train)
print("Training XGBoost (Robust Engine)...")
model_xgb.fit(X_train, y_train)

# --- 7. 加权集成 ---
# 因为做了 log1p，预测结果必须用 expm1 还原
lasso_preds = np.expm1(lasso.predict(X_test))
xgb_preds = np.expm1(model_xgb.predict(X_test))

# 0.5/0.5 混合，利用模型多样性降低方差
final_preds = (0.5 * lasso_preds) + (0.5 * xgb_preds)

# --- 8. 输出 Submission ---
submission = pd.DataFrame({'Id': test_id, 'SalePrice': final_preds})
submission.to_csv('submission.csv', index=False)
print("Success! File saved as submission.csv")