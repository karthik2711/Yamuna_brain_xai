from tensorflow.keras.models import load_model
from preprocessing.preprocess import load_images

def evaluate():
    model = load_model('trained_model.h5')
    X, y = load_images('data/sample_images')
    loss, acc = model.evaluate(X, y)
    print("Accuracy:", acc)
