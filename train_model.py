import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
from tqdm import tqdm

from utils.preprocess import load_tourism_data, feature_engineer_date

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Loading and merging data...")
raw = load_tourism_data('data/tourism_data.csv')
df = raw.copy()

print("Generating features...")
feature_rows = []
skipped = 0
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    feat = feature_engineer_date(df, row['site'], row['date'])

    if isinstance(feat, pd.Series):
        feat = feat.to_dict()
    elif isinstance(feat, pd.DataFrame):
        if not feat.empty:
            feat = feat.iloc[0].to_dict()
        else:
            skipped += 1
            continue
    elif isinstance(feat, (list, np.ndarray)):
        skipped += 1
        continue

    if isinstance(feat, dict):
        feat['site'] = row['site']  
        feat['tourists'] = row['tourists']
        feat['date'] = row['date']  
        feature_rows.append(feat)
    else:
        skipped += 1

print(f"Feature generation completed. Used: {len(feature_rows)} | Skipped: {skipped}")

X_full = pd.DataFrame(feature_rows)

print("Labelling crowd conditions...")
q25 = X_full['tourists'].quantile(0.25)
q75 = X_full['tourists'].quantile(0.75)

def label_cond(x):
    if x > q75: return 'Overcrowded'
    if x < q25: return 'Mild'
    return 'Suitable'

X_full['condition'] = X_full['tourists'].apply(label_cond)

X_full.to_csv("data/processed_data.csv", index=False)
print("Processed data saved to data/processed_data.csv")

print("Encoding target labels...")
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(X_full.pop('condition'))

print("Encoding feature 'site'...")
site_encoder = LabelEncoder()
X_full['site'] = site_encoder.fit_transform(X_full['site'])

X = X_full.drop(columns=['tourists', 'date'])  

print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training XGBoost classifier...")
clf = XGBClassifier(eval_metric='mlogloss', random_state=42)
clf.fit(X_train, y_train)

print("Model Evaluation:")
y_pred = clf.predict(X_test)
print(classification_report(target_encoder.inverse_transform(y_test), target_encoder.inverse_transform(y_pred)))

print("Saving model and encoders...")
joblib.dump(clf, 'models/tourism_classifier.pkl')
joblib.dump(target_encoder, 'models/target_encoder.pkl')
joblib.dump(site_encoder, 'models/site_encoder.pkl')
print("Model and encoders saved successfully.")

