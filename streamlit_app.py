import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocess the text input (same steps as in training)
def preprocess_text(text):
    import re
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    return text

# Streamlit UI
st.title('Sentiment Analysis using LLM')
user_input = st.text_input("Enter a sentence for sentiment analysis:")

if user_input:
    # Preprocess and vectorize input
    cleaned_input = preprocess_text(user_input)
    user_input_vectorized = vectorizer.transform([cleaned_input])

    # Get probability predictions
    probs = model.predict_proba(user_input_vectorized)
    max_prob = np.max(probs[0])  # Get highest probability score

    # Define a threshold for neutrality
    threshold = 0.6  # Adjust as needed

    if max_prob < threshold:
        sentiment = "Neutral"
    else:
        predicted_label = model.predict(user_input_vectorized)[0]
        sentiment = "Positive" if predicted_label == 4 else "Negative"

    # Display sentiment result
    st.write(f"The predicted sentiment for the input text is: **{sentiment}**")
