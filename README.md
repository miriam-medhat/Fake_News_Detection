# Fake_News_Detection

## 📁 Project Overview
This project classifies news articles as **REAL** or **FAKE** based on their text content. It uses machine learning with preprocessing, TF-IDF vectorization, and a Logistic Regression or SVM classifier.  
A Streamlit app is included to easily test news articles and display predictions with confidence scores.

---

## 🚀 Features
- Classifies news articles into **REAL** or **FAKE**  
- Implements **Logistic Regression** and **SVM** 
- Uses **TF-IDF vectorization** for feature extraction  
- Streamlit web app for real-time prediction  
- Bonus: Word cloud visualization of frequent terms in FAKE vs REAL news  

---

## 📊 Dataset
- **Source**: Real and Fake News Dataset
- Contains labels:  
  - REAL → genuine news articles  
  - FAKE → fabricated or misleading news  
- Balanced dataset with thousands of examples

---

## 🛠️ Preprocessing Steps
- Lowercasing  
- Removal of special characters, numbers, and punctuation  
- Tokenization  
- Stopword removal using **NLTK**  
- Lemmatization using **WordNet**  
- Feature extraction using **TF-IDF**    

---

## 📊 Model Summary
- **SVM (Support Vector Machine)**: main classifier for binary news detection  
- **Logistic Regression**: optional alternative classifier  

---

## 📌 About
Developed as part of an **NLP internship** to explore **fake news detection** using classical machine learning.  
- Includes a **Streamlit UI** for testing real-world articles
