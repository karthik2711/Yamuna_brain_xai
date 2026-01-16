import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y
