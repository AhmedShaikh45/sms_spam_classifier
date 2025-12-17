import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data (safe to call)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load trained model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title('Email / SMS Spam Classifier')

input_sms = st.text_input('Enter the message')

# Predict only when button is clicked
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)

        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. predict
        result = model.predict(vector_input)[0]

        # 4. display
        if result == 1:
            st.error("Spam 🚨")
        else:
            st.success("Not Spam ✅")

