import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer
model = load_model("lstm_sentiment_model__.keras")  # ✅ Correct way
with open("tokenizer__.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title('Movie Review Sentiment Analysis')
review = st.text_area('Enter a movie review:')

if st.button('Predict'):
    if review.strip():
        transformed_review = vectorizer.transform([review])
        prediction = model.predict(transformed_review)[0]
        sentiment = 'Positive' if prediction == 'positive' else 'Negative'
        st.write(f'The sentiment is: **{sentiment}**')
    else:
        st.write('Please enter a valid review.')
