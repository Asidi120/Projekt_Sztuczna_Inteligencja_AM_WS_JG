import cv2
import os
import numpy as np
from PIL import Image
import pickle
from Model_trenujacy import KlasyfikatorKNN

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

current_id = 0
label_ids = {}
y_labels = []
x_train = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Images")

def wczytaj_dane_treningowe(image_dir):
    x_train = []
    y_labels = []
    label_ids = {}
    current_id = 0

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                pil_image = Image.open(path).convert("L")
                size = (100, 100)
                final_image = pil_image.resize(size, Image.Resampling.LANCZOS)
                image_array = np.array(final_image, "uint8")
                x_train.append(image_array.flatten())
                y_labels.append(id_)

    return np.array(x_train), np.array(y_labels), label_ids

x_train, y_labels, label_ids = wczytaj_dane_treningowe(image_dir)

os.makedirs("pickles", exist_ok=True)
with open("pickles/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

knn = KlasyfikatorKNN(k=3)
knn.dopasuj(x_train, y_labels)
with open("pickles/knn-model.pickle", 'wb') as f:
    pickle.dump(knn, f)

print("Model KNN zapisany jako pickles/knn-model.pickle")

