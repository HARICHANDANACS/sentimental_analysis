# debug_model.py
import pickle

# Load the model and vectorizer
with open("emotion_classification_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Test the model with custom inputs
test_inputs = ["I am so happy today!", "I am feeling sad.", "This is so frustrating.", "I am hopeful for the future."]

# Transform the inputs using the same vectorizer
test_inputs_transformed = vectorizer.transform(test_inputs)

# Get the predictions for each test input
predictions = model.predict(test_inputs_transformed)

# Print predictions
for text, prediction in zip(test_inputs, predictions):
    print(f"Input: {text} => Predicted Emotion: {prediction}")
