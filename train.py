from model.model import build_model
from preprocessing.preprocess import preprocess_data

def train():
    X, y = preprocess_data('data/sample_data.csv')
    model = build_model(X.shape[1])
    
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
    model.save('trained_model.h5')
