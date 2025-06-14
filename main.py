import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("💻 LET'S COOK FAST 🧪")

DATASET_PATH = '/kaggle/input/burnout-datathon-ieeecsmuj/'
train = pd.read_csv(DATASET_PATH + 'train.csv')
val = pd.read_csv(DATASET_PATH + 'val.csv')
test = pd.read_csv(DATASET_PATH + 'test.csv')
sample_submission = pd.read_csv(DATASET_PATH + 'sample_submission.csv')

def cook_features(df):
    df = df.copy()
    if 'Avg_Speed_kmh' in df.columns:
        df['Speed_Squared'] = df['Avg_Speed_kmh'] ** 2
        df['Speed_Inv'] = 1 / (df['Avg_Speed_kmh'] + 0.001)
    if 'Grid_Position' in df.columns:
        df['Grid_Inv'] = 1 / (df['Grid_Position'] + 1)
    if 'Track_Temperature_Celsius' in df.columns:
        df['Track_Temp_Sq'] = df['Track_Temperature_Celsius'] ** 2
    if 'Avg_Speed_kmh' in df.columns and 'Grid_Position' in df.columns:
        df['Speed_Grid_Ratio'] = df['Avg_Speed_kmh'] / (df['Grid_Position'] + 1)
    return df

def fast_prep():
    tr, v, te = train.copy(), val.copy(), test.copy()
    for df in [tr, v, te]:
        if 'Penalty' in df.columns:
            df['Penalty'] = df['Penalty'].fillna(0)
        num = df.select_dtypes(include=[np.number]).columns
        for c in num:
            if c != 'Lap_Time_Seconds':
                df[c] = df[c].fillna(df[c].median())
    tr, v, te = cook_features(tr), cook_features(v), cook_features(te)
    cat_cols = tr.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([tr[col], v[col], te[col]]).astype(str)
        le.fit(all_vals)
        tr[col] = le.transform(tr[col].astype(str))
        v[col] = le.transform(v[col].astype(str))
        te[col] = le.transform(te[col].astype(str))
    drop_these = ['Unique ID', 'Rider_ID', 'Rider', 'Rider_name', 'Team_name', 'Bike_name', 'Shortname']
    for col in drop_these:
        for df in [tr, v, te]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
    return tr, v, te

print("🧼 Cleaning real quick...")
train_df, val_df, test_df = fast_prep()

X_train = train_df.drop('Lap_Time_Seconds', axis=1)
y_train = train_df['Lap_Time_Seconds']
X_val = val_df.drop('Lap_Time_Seconds', axis=1)
y_val = val_df['Lap_Time_Seconds']
X_test = test_df.copy()

print(f"📊 Features cooked: {X_train.shape[1]}")

params = {
    'n_estimators': 1500,
    'max_depth': 7,
    'learning_rate': 0.08,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 5,
    'reg_lambda': 5,
    'min_child_weight': 3,
    'gamma': 1,
    'random_state': 42,
    'tree_method': 'hist'
}

print("⚙️ Model go brrrrr...")
model = XGBRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

preds = model.predict(X_test)
rmse = mean_squared_error(y_val, model.predict(X_val), squared=False)
print(f"🏁 RMSE cooked: {rmse:.4f}")

submission = sample_submission.copy()
submission['Lap_Time_Seconds'] = preds
submission.to_csv('teamrocket_output.csv', index=False)

print("📦 yourteam_output.csv ready to ship 🚚")
💻 LET'S COOK FAST 🧪
🧼 Cleaning real quick...
📊 Features cooked: 47
⚙️ Model go brrrrr...
🏁 RMSE cooked: 5.0271
📦 yourteam_output.csv ready to ship 🚚
