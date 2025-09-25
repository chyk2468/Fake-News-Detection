import os
import io
import time
import pickle
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


MODEL_PATH = "model.pkl"


@st.cache_data(show_spinner=False)
def load_local_datasets(true_csv_path: str, fake_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    true_df = pd.read_csv(true_csv_path)
    fake_df = pd.read_csv(fake_csv_path)
    return true_df, fake_df


def _extract_text_column(df: pd.DataFrame) -> pd.Series:
    possible_cols = [
        "text",  # common Kaggle Fake/True dataset
        "content",
        "article",
        "body",
    ]
    for col in possible_cols:
        if col in df.columns:
            return df[col].astype(str)
    # Fallback: try the longest text-like column
    text_like_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_like_cols:
        raise ValueError("No suitable text column found in the dataset.")
    lengths = {c: df[c].astype(str).str.len().mean() for c in text_like_cols}
    best_col = max(lengths, key=lengths.get)
    return df[best_col].astype(str)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_df=0.9,
                    min_df=2,
                    max_features=100_000,
                ),
            ),
            ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
        ]
    )


def train_model(df_true: pd.DataFrame, df_fake: pd.DataFrame) -> Tuple[Pipeline, dict]:
    x_true = _extract_text_column(df_true)
    x_fake = _extract_text_column(df_fake)

    X = pd.concat([x_true, x_fake], ignore_index=True)
    y = np.concatenate([np.ones(len(x_true), dtype=int), np.zeros(len(x_fake), dtype=int)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "report": classification_report(y_val, y_pred, target_names=["Fake", "True"], output_dict=False),
    }
    return pipeline, metrics


@st.cache_resource(show_spinner=False)
def load_model_from_disk(model_path: str = MODEL_PATH) -> Optional[Pipeline]:
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def save_model_to_disk(model: Pipeline, model_path: str = MODEL_PATH) -> None:
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def predict_single(model: Pipeline, text: str) -> Tuple[int, float]:
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        label = int(np.argmax(proba))
        confidence = float(np.max(proba))
    else:
        label = int(model.predict([text])[0])
        confidence = 0.5
    return label, confidence


def predict_batch(model: Pipeline, texts: pd.Series) -> pd.DataFrame:
    texts = texts.fillna("").astype(str)
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(texts)
        labels = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)
    else:
        labels = model.predict(texts)
        confidences = np.full(shape=len(texts), fill_value=0.5)
    return pd.DataFrame({"prediction": labels, "confidence": confidences})


def app_title():
    st.title("Fake News Detection - Streamlit")
    st.caption("Automatically trains on local True/Fake CSVs (or loads saved model), then classifies text.")


def sidebar_info():
    with st.sidebar:
        st.header("About")
        st.write("This app auto-loads `model.pkl` or trains on local `True.csv` and `Fake.csv`. Use the text box to classify news as Fake or Real.")
        st.markdown("**Labels**: 1 = Real, 0 = Fake")
        if os.path.exists(MODEL_PATH):
            st.success("Existing model found: model.pkl")
        else:
            st.info("No saved model yet. App will train on startup if CSVs exist.")


def auto_prepare_model() -> Optional[Pipeline]:
    model = load_model_from_disk()
    if model is not None:
        st.info("Loaded existing model from disk.")
        return model
    true_exists = os.path.exists("True.csv")
    fake_exists = os.path.exists("Fake.csv")
    if not (true_exists and fake_exists):
        st.error("Missing local CSVs. Place `True.csv` and `Fake.csv` next to app.py, then reload.")
        return None
    with st.spinner("Training on local True.csv and Fake.csv..."):
        try:
            true_df, fake_df = load_local_datasets("True.csv", "Fake.csv")
            model, metrics = train_model(true_df, fake_df)
        except Exception as e:
            st.error(f"Training failed: {e}")
            return None
    st.success(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")
    try:
        save_model_to_disk(model)
        st.toast("Model saved to model.pkl", icon="✅")
    except Exception as e:
        st.warning(f"Could not save model: {e}")
    return model


def prediction_section(model: Optional[Pipeline]):
    st.subheader("Predict")
    if model is None:
        st.info("Model not available. Ensure `True.csv` and `Fake.csv` are present, then reload.")
        return
    text = st.text_area("Enter news text", height=200)
    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Classifying..."):
                label, conf = predict_single(model, text)
            label_name = "Real" if label == 1 else "Fake"
            st.metric("Prediction", label_name, delta=f"confidence {conf:.2f}")


def main():
    app_title()
    sidebar_info()
    model = auto_prepare_model()
    prediction_section(model)


if __name__ == "__main__":
    main()


