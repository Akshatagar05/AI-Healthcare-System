import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def train_history_model(username):
    conn = sqlite3.connect("database.db")

    df = pd.read_sql_query(
        "SELECT message, response FROM chats WHERE username=?",
        conn,
        params=(username,)
    )

    conn.close()

    if len(df) < 5:
        return None, None

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['message'])
    y = df['response']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model, vectorizer

def predict_with_history(model, vectorizer, user_input):
    X = vectorizer.transform([user_input])
    return model.predict(X)[0]