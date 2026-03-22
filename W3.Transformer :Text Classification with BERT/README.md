# 🎬 BERT for IMDb Sentiment Classification

## 📌 Overview

This project implements a **sentiment classification model** using a **Transformer-based architecture (BERT)** to classify movie reviews from the IMDb dataset as **positive** or **negative**.

The goal is to leverage pre-trained language models to achieve high accuracy in natural language understanding tasks.

---

## 🎯 Objectives

* Build a sentiment analysis model using **BERT (Transformer)**
* Classify IMDb movie reviews into:

  * Positive 😊
  * Negative 😞
* Achieve **≥ 85% accuracy** on the test dataset

---

## 📊 Dataset

* **Dataset:** IMDb Movie Reviews
* **Source:** Hugging Face `datasets` library
* **Size:**

  * 25,000 training samples
  * 25,000 testing samples

---

## 🧠 Model

* **Architecture:** BERT (Bidirectional Encoder Representations from Transformers)
* **Framework:** Hugging Face Transformers + PyTorch
* **Task:** Binary Text Classification

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* Matplotlib

---

## 🛠️ Installation

Run the following in your environment (Colab recommended):

```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install matplotlib scikit-learn
```

---

## 🚀 Workflow

1. **Load Dataset**

   ```python
   dataset = load_dataset("imdb")
   ```

2. **Data Preprocessing**

   * Tokenization using BERT tokenizer
   * Padding & truncation

3. **Model Training**

   * Fine-tune pre-trained BERT
   * Train on labeled IMDb dataset

4. **Evaluation**

   * Measure accuracy on test set
   * Target ≥ 85%

---

## 📈 Expected Results

* High accuracy in sentiment classification
* Strong performance due to contextual understanding from BERT
* Better results compared to traditional ML models

---

## 🏁 Learning Outcomes

Through this project, I gained:

* Understanding of **Transformer architecture (BERT)**
* Experience with **Hugging Face ecosystem**
* Knowledge of **text preprocessing & tokenization**
* Hands-on practice with **fine-tuning pre-trained models**
* Insight into **NLP classification workflows**

---

## 🚧 Future Improvements

* Add confusion matrix and detailed metrics (Precision, Recall, F1)
* Experiment with other models (RoBERTa, DistilBERT)
* Hyperparameter tuning for better performance
* Deploy as an API or web app

---

## 📎 Notebook

👉 https://colab.research.google.com/drive/1XoJ6vejRdYtpgLti8MRhRkiKAJlJ-16m#scrollTo=PmAwiN-9zgtX

---

## 👤 Author

**Chih-Hui**

