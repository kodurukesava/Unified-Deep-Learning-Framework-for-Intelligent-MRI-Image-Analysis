import os
import json
import numpy as np


def load_class_names(path):

    with open(path, "r") as file:
        data = json.load(file)

    return {
        int(key): value
        for key, value in data.items()
    }


def ensure_directory(path):

    os.makedirs(path, exist_ok=True)


def softmax_percentage(probabilities):

    return [
        round(float(probability) * 100, 2)
        for probability in probabilities
    ]


def get_top_prediction(predictions, class_names):

    index = int(np.argmax(predictions))

    confidence = float(predictions[index])

    return (
        class_names[index],
        confidence
    )