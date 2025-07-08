# 🌍 Multilingual Language Detection using Machine Learning

This project is a web-based application that identifies the **language of a given text** using a **machine learning model** trained on character-level features. It combines **Flask** (for the backend), **scikit-learn** (for ML), and **Tailwind CSS** (for the frontend).

## 📌 Features

- Detects languages from short text inputs.
- Supports over 20 languages including English, French, Hindi, Spanish, Chinese, Arabic, and more.
- Frontend built with HTML + Tailwind CSS.
- Backend built with Flask REST API.
- Trained using Logistic Regression and TF-IDF vectorization (character-level).

## 🧠 Model Overview

| Component           | Technique                       |
|--------------------|----------------------------------|
| Feature Extraction | `TfidfVectorizer` (char-level n-grams 1–2) |
| Classifier         | `LogisticRegression`             |
| Accuracy           | ~98% on test set                 |

## 🗂 Project Structure

```
language-detection/
├── app/
│   ├── main.py             # Flask app serving predictions
│   ├── model.joblib        # Trained ML model
├── data/
│   ├── dataset.csv         # Dataset (Text, Language)
├── frontend/
│   ├── index.html          # Web interface
├── train.py                # Script to train and save the model
├── predict.py              # CLI-based prediction script
└── README.md
```

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/language-detection.git
cd language-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

### 4. Run the Web Application

```bash
cd app
python main.py
```

## 🌐 Web Interface

- Enter any sentence in the textbox.
- Click on **Detect Language**.
- Get immediate feedback showing the predicted language.

## 🧪 Example (CLI usage)

```bash
python predict.py
```

## 📊 Dataset

- Source: [Kaggle Language Dataset](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst)

## 📈 Model Evaluation (Partial)

| Language  | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| English   | 0.81      | 0.99   | 0.89     |
| French    | 0.96      | 0.99   | 0.97     |
| Hindi     | 1.00      | 0.99   | 0.99     |
| Russian   | 0.98      | 1.00   | 0.99     |
| Tamil     | 1.00      | 0.99   | 1.00     |

> Accuracy: **98%**

## 📌 Future Enhancements

- Add support for **code-mixed** and **multilingual** inputs.
- Integrate with **Transformer-based models** (e.g., BERT, XLM-R).
- Deploy using **Docker** or **Heroku**.
- Add UI for confidence scores and error handling.


## 📄 License

This project is for academic purposes. Contact the author for licensing queries.
