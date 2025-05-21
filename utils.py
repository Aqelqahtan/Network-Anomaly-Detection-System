import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_kdd(file_path):
    df = pd.read_csv(file_path)

    # Encode categorical columns
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Convert label to binary anomaly class
    df["anomaly"] = df["label"].apply(lambda x: 1 if x == "normal" else 0)
    df.drop(columns=["label"], inplace=True)

    return df

def prepare_features(df):
    X = df.drop(columns=["anomaly"])
    y = df["anomaly"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def save_train_test_split(df, train_path="data/train.csv", test_path="data/test.csv", test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["anomaly"], random_state=42)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"✅ Saved train ({len(train_df)}) and test ({len(test_df)}) to CSV.")
