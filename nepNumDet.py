import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "C:/Users/saimo/Downloads/images"
CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]

# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
#         plt.imshow(img_array, cmap="gray")
#         plt.show()
#         break
#     break

IMG_SHAPE = 50
#img_array = np.array([])

#new_array = cv2.resize(img_array, (IMG_SHAPE, IMG_SHAPE))
# plt.imshow(new_array, cmap='gray')
# plt.show()

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SHAPE, IMG_SHAPE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    
create_training_data()

random.shuffle(training_data)

X = [] 
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SHAPE, IMG_SHAPE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)