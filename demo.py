import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, TFAutoModelForMultipleChoice

tokenizer = AutoTokenizer.from_pretrained("vanhuyen/demo")
model = TFAutoModelForMultipleChoice.from_pretrained("vanhuyen/demo")

def predict_answer(question, choices):
    """
    Predicts the answer to a multiple-choice question using a trained model.
    """
    reverse_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    first_sentences = [question] * len(choices)
    second_sentences = choices

    tokenized = tokenizer(first_sentences, second_sentences, padding="longest", return_tensors="np")
    tokenized = {key: np.expand_dims(array, 0) for key, array in tokenized.items()}

    outputs = model(tokenized).logits
    logits = outputs[0]
    probabilities = softmax(logits)

    top_n = 1
    top_indices = np.argsort(probabilities)[::-1][:top_n]

    result = {}
    for i, idx in enumerate(top_indices):
        result[f"Choice {reverse_mapping.get(idx)}"] = probabilities[idx]

    return result