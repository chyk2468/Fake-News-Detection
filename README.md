<div align="center">

# 📰 Fake News Detection System

### Real-Time News Authenticity Classifier powered by ML + Streamlit

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TF-IDF](https://img.shields.io/badge/TF--IDF-NLP-blueviolet?style=for-the-badge)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A **Streamlit web application** that detects fake news in real-time using a **Logistic Regression + TF-IDF** pipeline trained on 44,000+ labeled news articles.

</div>

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Auto-Training** | Trains model on `True.csv` & `Fake.csv` if no saved model exists |
| 💾 **Model Persistence** | Saves trained model as `model.pkl` for instant reload on next launch |
| ⚡ **Real-Time Prediction** | Classifies any pasted news article instantly |
| 📊 **Confidence Score** | Shows how confident the model is in its prediction |
| 🎨 **Interactive UI** | Clean Streamlit interface with sidebar info and status messages |
| 🔠 **NLP Pipeline** | TF-IDF vectorization → Logistic Regression classification |

---

## 🖥️ App Preview

**Fake News Detected:**

<img width="1919" height="1073" alt="Fake prediction output" src="https://github.com/user-attachments/assets/69fefdcb-21b9-409a-9ca6-9aeda07f76a9" />

---

**Real News Detected:**

<img width="1919" height="1079" alt="Real prediction output" src="https://github.com/user-attachments/assets/9f2e332f-592c-4934-8d7a-0783825c38f9" />

---

## ⚙️ How It Works

```
User pastes news article text
           │
           ▼
  ┌──────────────────────┐
  │  model.pkl exists?   │
  └────────┬─────────────┘
      No ──┤──── Yes
           │         │
           ▼         ▼
  ┌──────────────┐  ┌───────────────┐
  │ Load         │  │ Load saved    │
  │ True.csv &   │  │ model.pkl     │
  │ Fake.csv     │  └──────┬────────┘
  └──────┬───────┘         │
         ▼                 │
  ┌──────────────┐         │
  │ TF-IDF       │         │
  │ Vectorize +  │         │
  │ Train LR     │         │
  └──────┬───────┘         │
         ▼                 │
  ┌──────────────┐         │
  │  Save        │         │
  │  model.pkl   │         │
  └──────┬───────┘         │
         └────────┬─────────┘
                  ▼
         ┌─────────────────┐
         │ Predict: FAKE   │
         │   or  REAL      │
         │ + Confidence %  │
         └─────────────────┘
```

---

## 📊 Dataset

| File | Content | Size |
|------|---------|------|
| `True.csv` | Genuine news articles | ~21,000 articles |
| `Fake.csv` | Fake / misinformation articles | ~23,000 articles |

**Source:** [ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) — widely used benchmark for fake news classification research.

---

## 🧠 Model Architecture

### NLP Pipeline
```
Raw Text Input
      │
      ▼
TF-IDF Vectorizer
  ├── max_features = 5000
  ├── stop_words = 'english'
  └── ngram_range = (1, 2)
      │
      ▼
Logistic Regression Classifier
  ├── solver = 'lbfgs'
  ├── max_iter = 1000
  └── Output: [0 = Fake, 1 = Real] + Probabilities
```

**Why Logistic Regression + TF-IDF?**
- ✅ Fast to train on large text corpora
- ✅ Highly interpretable predictions
- ✅ Strong baseline accuracy for binary text classification
- ✅ Lightweight — no GPU required

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/chyk2468/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the App

```bash
streamlit run app.py
```

> 💡 The app will automatically train the model on first launch using `True.csv` and `Fake.csv`. This takes ~30–60 seconds. Subsequent launches load instantly from `model.pkl`.

---

## 📑 Example

**Input:**
```
Breaking: Celebrity endorses miracle cure for COVID-19!
Scientists baffled as underground lab reveals Earth is flat.
```

**Output:**
```
Prediction  :  🔴 FAKE
Confidence  :  91.3%
```

**Input:**
```
Federal Reserve raises interest rates by 0.25% amid inflation concerns.
```

**Output:**
```
Prediction  :  🟢 REAL
Confidence  :  87.6%
```

---

## 📁 Project Structure

```
📦 Fake-News-Detection/
├── 📓 Fake News Spam Detection System.ipynb   # Exploratory notebook
├── 🐍 app.py                                  # Streamlit application
├── 📊 True.csv                                # Real news dataset
├── 📊 Fake.csv                                # Fake news dataset
├── 🤖 model.pkl                               # Saved trained model (auto-generated)
├── 📋 requirements.txt                        # Python dependencies
└── 📄 README.md                               # Project documentation
```

---

## 📦 Requirements

```
streamlit==1.38.0
pandas==2.2.2
numpy==2.0.1
scikit-learn==1.5.2
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of all predicted fakes, how many were actually fake |
| **Recall** | Of all actual fakes, how many were correctly caught |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confidence** | Model's predicted probability shown in the UI |

---

## 🛠️ Technologies Used

<div align="center">

| Category | Tool |
|----------|------|
| **Language** | Python 3.8+ |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **NLP** | Scikit-Learn TF-IDF |
| **Classifier** | Logistic Regression |
| **Model Persistence** | Pickle (`model.pkl`) |

</div>

---

## 👤 Author

**Yashwant Kumar Chitchula**  
B.Tech CSE (AI & ML) — VIT Chennai

[![GitHub](https://img.shields.io/badge/GitHub-chyk2468-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/chyk2468)

---

<div align="center">

⭐ **Found this useful? Drop a star!** ⭐

</div>
