import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or however many cores you want
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization & Preprocessing
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# Advanced Models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# 1. Load and Setup Data
# --------------------------------------------------------------------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ID = test['Id']

# Drop ID
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# 2. Advanced Preprocessing
# --------------------------------------------------------------------------------

# Remove Outliers (Crucial for regression)
# Documentation suggests outliers in GrLivArea > 4000
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

# Log Transform Target Variable
# Evaluation metric is RMSE of Log(Price), so we train on Log(Price)
train["SalePrice"] = np.log1p(train["SalePrice"])
y_train = train.SalePrice.values
ntrain = train.shape[0]
ntest = test.shape[0]

# Concatenate for consistent feature engineering
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

# 3. Imputation (Handling Missing Values)
# --------------------------------------------------------------------------------
# Meaningful NAs
for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']:
    all_data[col] = all_data[col].fillna("None")

# Numerical NAs fill with 0
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 
            'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']:
    all_data[col] = all_data[col].fillna(0)

# LotFrontage: Fill with median of neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

# Mode filling for others
for col in ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional']:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# 4. Feature Engineering
# --------------------------------------------------------------------------------

# Convert categorical-looking numericals to string
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

# Create "Total Square Footage" feature (Strong Correlation with Price)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# Fix Skewness in numerical features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = skewed_feats[abs(skewed_feats) > 0.75]
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

# One-Hot Encoding
all_data = pd.get_dummies(all_data)

# Robust Scaling (Critical for Neural Networks)
scaler = RobustScaler()
all_data_scaled = scaler.fit_transform(all_data)

X_train = all_data_scaled[:ntrain]
X_test = all_data_scaled[ntrain:]

print(f"Data shape: {X_train.shape}")

# 5. Defining Advanced Models
# --------------------------------------------------------------------------------

# A. XGBoost
xgb_model = xgb.XGBRegressor(
    colsample_bytree=0.4603, gamma=0.0468, 
    learning_rate=0.05, max_depth=3, 
    min_child_weight=1.7817, n_estimators=2200,
    reg_alpha=0.4640, reg_lambda=0.8571,
    subsample=0.5213, random_state=7, n_jobs=-1, verbose=0
)

# B. LightGBM
lgb_model = lgb.LGBMRegressor(
    objective='regression', num_leaves=5,
    learning_rate=0.05, n_estimators=720,
    max_bin=55, bagging_fraction=0.8,
    bagging_freq=5, feature_fraction=0.2319,
    feature_fraction_seed=9, bagging_seed=9,
    min_data_in_leaf=6, min_sum_hessian_in_leaf=11, verbose=-1
)

# C. CatBoost
cat_model = CatBoostRegressor(
    iterations=3000, learning_rate=0.01,
    depth=5, l2_leaf_reg=3,
    loss_function='RMSE', verbose=0, random_state=42
)

# D. Neural Network (Keras)
def build_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(1)) # Regression output
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 6. Training
# --------------------------------------------------------------------------------

print("Training XGBoost...")
xgb_model.fit(X_train, y_train)
xgb_pred = np.expm1(xgb_model.predict(X_test))

print("Training LightGBM...")
lgb_model.fit(X_train, y_train)
lgb_pred = np.expm1(lgb_model.predict(X_test))

print("Training CatBoost...")
cat_model.fit(X_train, y_train)
cat_pred = np.expm1(cat_model.predict(X_test))

print("Training Neural Network...")
nn_model = build_nn_model(X_train.shape[1])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Using a simple validation split for early stopping
nn_model.fit(X_train, y_train, validation_split=0.2, epochs=150, batch_size=32, 
             callbacks=[early_stop], verbose=0)
nn_pred_log = nn_model.predict(X_test)
nn_pred = np.expm1(nn_pred_log).flatten()

# 7. Blending (Ensemble)
# --------------------------------------------------------------------------------

# Weights are assigned based on typical model performance and diversity
# Boosters usually handle this data best, NN adds diversity
weighted_prediction = (0.35 * cat_pred) + \
                      (0.25 * xgb_pred) + \
                      (0.25 * lgb_pred) + \
                      (0.15 * nn_pred)

# 8. Submission
# --------------------------------------------------------------------------------
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = weighted_prediction
sub.to_csv('submission2_1.csv', index=False)

print("Advanced Submission file created successfully!")

# Optional: Print sample predictions
print("\nSample Predictions (Top 5):")
print(sub.head())