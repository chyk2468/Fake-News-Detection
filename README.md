# Fake News Detection

**Classify news text with TF-IDF features and a Logistic Regression model.**

[![GitHub](https://img.shields.io/badge/GitHub-Source-24292f?logo=github)](https://github.com/chyk2468/Fake-News-Detection)

## 1. Project Title & Tagline

Classify news text with TF-IDF features and a Logistic Regression model.

## 2. Problem Statement

News articles and social posts can spread misleading claims quickly. This project demonstrates a text-classification pipeline that predicts a real/fake label from article text.

## 3. Project Preview

Add a real screenshot, GIF, or public demo here after recording the current application.

[Interactive architecture explorer](architecture.html)

## 4. Overview

The application prepares text as TF-IDF features and uses a Logistic Regression classifier trained from `True.csv` and `Fake.csv`. The Streamlit UI presents the predicted label and a model score.

## 5. Key Features

- Classifies pasted news text.
- Uses TF-IDF text features.
- Trains or loads a Logistic Regression model.
- Displays the predicted class and a confidence-style score.

## 6. Architecture / Workflow

![Animated Fake News Detection workflow](readme-architecture.svg)

The diagram animates the flow between stages. Open the [interactive architecture explorer](architecture.html) to select each stage and read its role.

## 7. Tech Stack

Python, Pandas, scikit-learn, TF-IDF, Logistic Regression, Streamlit

## 8. Installation & Setup

```bash
git clone https://github.com/chyk2468/Fake-News-Detection.git
cd Fake-News-Detection
python -m venv .venv
pip install -r requirements.txt
```

## 9. Usage

1. Run `streamlit run app.py`.
2. Paste the news text into the input field.
3. Submit it and inspect the predicted label and score.
4. Treat the result as a model prediction, not a fact-checking verdict.

## 10. Project Structure

- `app.py`: Streamlit application
- `Fake.csv`: fake-news training examples
- `True.csv`: real-news training examples
- `Fake News Spam Detection System.ipynb`: notebook exploration
- `requirements.txt`: dependencies

## 11. Results & Demo

The repository contains labelled CSV datasets and a Jupyter notebook. Add measured precision, recall, F1-score, dataset split details, and a screenshot after running a reproducible evaluation.

## 12. Future Enhancements

- Report precision, recall, F1-score, and a confusion matrix.
- Test on recent and out-of-domain news.
- Add explainability for influential words.
- Make the interface clear that predictions can be wrong.

## 13. License

Check for a repository `LICENSE` file before redistributing the code or datasets.
