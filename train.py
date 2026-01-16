from preprocessing.preprocess import load_images
from model.model import build_model
from tensorflow.keras.optimizers import Adam

def train():
    X, y = load_images('data/sample_images')
    model = build_model(X.shape[1:], len(set(y)))
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=16)
    model.save('trained_model.h5')
