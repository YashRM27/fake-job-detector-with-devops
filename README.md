# 🔍 Fake Job Posting Detector

A machine learning web app that predicts whether a job posting is **Real or Fake** using NLP and classification.

**Live Demo →** *(add your Streamlit Cloud URL here)*

---

## Problem Statement

Job scams cost victims thousands of dollars annually. Fake postings often use vague language, skip key details (salary, company info), and include red-flag phrases like "urgent hiring" or "no experience needed". This project builds a binary classifier to flag such postings before job seekers engage.

---

## Dataset

**EMSCAD – Real or Fake Job Postings**  
Source: [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

| Attribute | Value |
|---|---|
| Total records | 17,880 |
| Features | 18 |
| Target | `fraudulent` (0 = Real, 1 = Fake) |
| Class imbalance | ~95.2% Real / 4.8% Fake |

---

## Approach

### Feature Engineering
- Combined `title + company_profile + description + requirements + benefits` → single TF-IDF corpus
- TF-IDF vectorisation (5,000 features, unigram + bigram, sublinear TF)
- Missingness flags: fields like `salary_range`, `company_profile` being null are predictive signals
- Text length features: `desc_len`, `title_len`, `text_len`
- Red flag keyword count: 14 scam-associated phrases
- Label-encoded categoricals: `employment_type`, `required_experience`, `required_education`

### Imbalance Handling
- **SMOTE** applied inside `imblearn.pipeline.Pipeline` to prevent data leakage into validation folds

### Models Trained
| Model | Fake Recall | Fake F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| **XGBoost (tuned)** | **—** | **—** | **—** |

*(Fill in your actual numbers after running the notebook)*

### Tuning
- `RandomizedSearchCV` over XGBoost hyperparameters
- Optimised for **F1 score on the fake class** (Recall-weighted to minimise missed scams)

---

## Project Structure

```
fake-job-detector/
├── notebooks/
│   ├── 01_eda_feature_engineering.ipynb
│   └── 02_modelling.ipynb
├── model/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── structured_features.pkl
├── app.py                  # Streamlit app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/fake-job-detector.git
cd fake-job-detector

pip install -r requirements.txt

# Run notebooks first to generate model/ artefacts
jupyter notebook

# Then launch the app
streamlit run app.py
```

---

## Deployment

Deployed on **Streamlit Community Cloud** (free).

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Deploy

**Important:** The `model/` folder (with `.pkl` files) must be committed to GitHub for the app to load on Streamlit Cloud.

---

## App Features

- Paste any job description text → get Real / Fake prediction
- Confidence score (probability from model)
- 🚩 Red flag keyword detection with highlighted phrases
- Signal breakdown: logo, questions, missingness indicators
- Works for any job board (LinkedIn, Indeed, Naukri, etc.)

---

## Tech Stack

`Python` `Pandas` `Scikit-learn` `XGBoost` `imbalanced-learn` `SciPy` `Streamlit`

---

## Author

**Yash Mavare**  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)
