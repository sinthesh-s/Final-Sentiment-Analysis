import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("lstm_sentiment_model__.keras")  # Keras LSTM model
with open("tokenizer__.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Sequence padding length (should match what was used during training)
MAX_LEN = 200

# Streamlit UI
st.title('Movie Review Sentiment Analysis')
review = st.text_area('Enter a movie review:')

if st.button('Predict'):
    if review.strip():
        # Convert text to sequence and pad
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Predict sentiment
        pred = model.predict(padded)[0]  # e.g., [0.2, 0.6, 0.2]
        sentiment_index = np.argmax(pred)
        sentiment_label = ["Negative", "Neutral", "Positive"][sentiment_index]
        confidence = float(pred[sentiment_index])

        # Display result
        st.write(f'The sentiment is: **{sentiment_label}**')
        st.write(f'Confidence: **{confidence:.2f}**')
    else:
        st.write('Please enter a valid review.')
