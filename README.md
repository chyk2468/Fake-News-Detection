## Fake News Detection - Streamlit App

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Place `True.csv` and `Fake.csv` in the same folder as `app.py` (or upload them via the UI). The app trains a TF-IDF + Logistic Regression model, saves it as `model.pkl`, and supports single-text and CSV batch predictions.


