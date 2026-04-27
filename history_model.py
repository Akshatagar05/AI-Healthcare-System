import sqlite3
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MIN_SAMPLES = 5

def train_history_model(username: str):
    """
    Train a RandomForest model on the user's past chat messages.
    Returns (model, vectorizer) or (None, None) if insufficient data.
    """
    try:
        conn = sqlite3.connect("database.db")
        df = pd.read_sql_query(
            "SELECT message, disease FROM chats WHERE username=? AND disease NOT IN ('Triage Phase', 'Triage / General')",
            conn, params=(username,)
        )
        conn.close()

        if len(df) < MIN_SAMPLES:
            return None, None

        # Drop rows with missing values
        df = df.dropna(subset=["message", "disease"])
        df = df[df["disease"].str.strip() != ""]

        if len(df) < MIN_SAMPLES:
            return None, None

        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        X = vectorizer.fit_transform(df["message"])

        le = LabelEncoder()
        y = le.fit_transform(df["disease"])

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        # Store label encoder on the model for decoding
        clf.label_encoder_ = le

        return clf, vectorizer

    except Exception as e:
        logger.warning(f"History model training failed for {username}: {e}")
        return None, None


def predict_with_history(model, vectorizer, user_input: str) -> str:
    """
    Predict the most likely condition based on the user's message and past history.
    """
    try:
        X = vectorizer.transform([user_input])
        pred_encoded = model.predict(X)[0]
        # Decode label back to string
        label = model.label_encoder_.inverse_transform([pred_encoded])[0]
        return label
    except Exception as e:
        logger.warning(f"History prediction failed: {e}")
        return "No prior history pattern."