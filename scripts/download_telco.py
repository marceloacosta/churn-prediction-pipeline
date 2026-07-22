"""Download the IBM Telco Customer Churn dataset."""
import os
import urllib.request

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEST = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

if not os.path.exists(DEST):
    print("Downloading IBM Telco Customer Churn dataset...")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {DEST}")
else:
    print(f"Already exists: {DEST}")

df = pd.read_csv(DEST)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(df.head(3))
