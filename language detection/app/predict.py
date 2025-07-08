import joblib
import os

# Load the saved model with relative path
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

def predict_language(text):
    """
    Predict the language of the given text.
    
    Args:
        text (str): Input text.

    Returns:
        str: Predicted language.
    """
    return model.predict([text])[0]

# Example usage
if __name__ == "__main__":
    sample_text = "Bonjour, comment allez-vous?"
    print("Predicted language:", predict_language(sample_text))
