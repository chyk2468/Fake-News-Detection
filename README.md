# 📰 Fake News Detection - Streamlit App

This project is a **Streamlit-based Fake News Detection System** that classifies news articles as either **Fake** or **Real**. It automatically trains a **Logistic Regression + TF-IDF pipeline** on local datasets (`True.csv` and `Fake.csv`) or loads a pre-trained `model.pkl`.

---

## 📁 Project Files

* `True.csv` — Dataset containing genuine news articles.
* `Fake.csv` — Dataset containing fake news articles.
* `app.py` — Main Streamlit application file.
* `model.pkl` — Saved trained model (auto-created after training).
* `README.md` — Project documentation.

---

## ⚙️ Features

* Automatically detects and trains model on `True.csv` & `Fake.csv` if no model exists.
* Uses **TF-IDF Vectorization** and **Logistic Regression** for classification.
* Saves the trained model as `model.pkl` for faster reloading.
* Predicts news authenticity in real-time from text input.
* Displays prediction label (`Fake` or `Real`) and model confidence.
* Provides sidebar info and app status messages.

---

## 🚀 Setup & Run

### 1️⃣ Clone the repo

```bash
git clone https://github.com/chyk2468/fake-news-detection.git
cd fake-news-detection
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit app

```bash
streamlit run app.py
```

---

## 📊 How It Works

1. On startup:

   * If `model.pkl` exists → loads the model.
   * Else, checks for `True.csv` and `Fake.csv`, then trains and saves the model.
2. User inputs a news article in the text box.
3. The model predicts **Fake (0)** or **Real (1)** with a confidence score.
4. Displays results interactively.

---

## 📑 Example Usage

Input:

```
Breaking: Celebrity endorses miracle cure for COVID-19!
```

Output:

```
Prediction: Fake
Confidence: 0.91
```
<img width="1919" height="1073" alt="image" src="https://github.com/user-attachments/assets/69fefdcb-21b9-409a-9ca6-9aeda07f76a9" />

---

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/9f2e332f-592c-4934-8d7a-0783825c38f9" />

---

## 🔍 Project Structure

```
fake-news-detection/
│
├── app.py
├── True.csv
├── Fake.csv
├── model.pkl
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

Create a `requirements.txt` with:

```
streamlit
pandas
numpy
scikit-learn
```

---

## ⚡ Author

**Yashwant Kumar Chitchula**
B.Tech CSE (AI & ML), VIT Chennai
