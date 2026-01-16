import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.preprocess import preprocess_data

def evaluate():
    model = load_model('trained_model.h5')
    X, y = preprocess_data('data/sample_data.csv')
    
    loss, acc = model.evaluate(X, y)
    print("Accuracy:", acc)
