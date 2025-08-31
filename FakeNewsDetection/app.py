import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load SVM Model & Vectorizer
model = pickle.load(open("svm_model.pkl", "rb"))  # SVM model
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()  # lowercasing
    text = re.sub(r'[^a-z\s]', '', text)  # remove special chars, numbers, punctuation
    tokens = [word for word in text.split() if word not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # lemmatize
    return ' '.join(tokens)

# Prediction function
def predict_news(text):
    clean_text = preprocess(text)
    text_vector = vectorizer.transform([clean_text])
    
    # SVM prediction
    pred = model.predict(text_vector)[0]
    label = "REAL" if pred == "REAL" else "FAKE"
    return label

# UI
st.markdown("<h1 style='text-align:center; font-weight:bold;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("Paste a news article below and check if it's **Real** or **Fake**.")

user_input = st.text_area("Enter news article text here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        label = predict_news(user_input)
        color = "green" if label == "REAL" else "red"
        st.markdown(f"**Prediction:** <span style='color:{color}; font-weight:bold;'>{label}</span>", unsafe_allow_html=True)
