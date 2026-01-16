import cv2
import numpy as np
import os

def load_images(folder, img_size=(224,224)):
    images = []
    labels = []
    for label, cls in enumerate(os.listdir(folder)):
        cls_path = os.path.join(folder, cls)
        for img in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img)
            im = cv2.imread(img_path)
            im = cv2.resize(im, img_size)
            im = im / 255.0
            images.append(im)
            labels.append(label)
    return np.array(images), np.array(labels)
