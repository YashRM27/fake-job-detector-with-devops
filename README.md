# 🔍 Fake Job Posting Detector — Cloud-Native ML Service

A machine learning web app that predicts whether a job posting is **Real or Fake** using NLP and classification.

**Live Demo →** [Streamlit App](https://yashRM27-fake-job-detector.streamlit.app)

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
- **SMOTE** applied during training
- XGBoost `scale_pos_weight` set to 19:1 (negative/positive ratio) for native imbalance handling

### Models Trained
| Model | Fake Recall | Fake F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| **XGBoost (final)** | **0.88** | **0.76** | **0.98** |

### Tuning
- `RandomizedSearchCV` over XGBoost hyperparameters
- Optimised for **F1 score on the fake class** (Recall-weighted to minimise missed scams)

---

## Project Structure

```
fake-job-detector-with-devops/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── model/
├── notebooks/
├── k8s/
│   ├── deployment.yml
│   └── service.yml
├── ansible/
│   └── playbook.yml
├── .github/
│   └── workflows/
│       └── docker-build.yml
└── README.md
```

---

## DevOps Architecture

| Layer | Tool | Purpose |
|---|---|---|
| Containerisation | Docker | Package app + dependencies |
| CI/CD | GitHub Actions | Auto build & push on every commit |
| Orchestration | Kubernetes | 2 replica deployment with health checks |
| Provisioning | Ansible | One-command environment setup |
| Registry | Docker Hub | Public image repository |

## Quick Start

### Option 1 — Docker
```bash
docker pull yashrm27/fake-job-detector:latest
docker run -p 8501:8501 yashrm27/fake-job-detector:latest
```

### Option 2 — Ansible
```bash
ansible-playbook ansible/playbook.yml
```

### Option 3 — Kubernetes
```bash
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
minikube service fake-job-detector-service
```

---

## Running Locally

```bash
git clone https://github.com/YashRM27/fake-job-detector.git
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

## Docker

Pull and run the app locally using Docker:

```bash
docker pull yashrm27/fake-job-detector
docker run -p 8501:8501 yashrm27/fake-job-detector
```

Then open **http://localhost:8501** in your browser.

Docker Hub: https://hub.docker.com/r/yashrm27/fake-job-detector

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
[LinkedIn](https://linkedin.com/in/yashmavare) · [GitHub](https://github.com/YashRM27)