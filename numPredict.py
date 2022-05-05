import cv2
import tensorflow as tf
import os

CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]
IMG_SHAPE = 50

def prepare(filepath):
	img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	new_array = cv2.resize(img_array, (IMG_SHAPE, IMG_SHAPE))
	return new_array.reshape(-1, IMG_SHAPE, IMG_SHAPE, 1)

model = tf.keras.models.load_model("64x2-CNN-3_64_1.model")
directory = "C:/Users/saimo/PycharmProjects/Untitled Folder/Tensorflow/workspace/numDet/testNumbers/"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        prediction = model.predict([prepare(f)])
        for index, val in enumerate(prediction[0]):
        	if val == 1:
        		p = "filename: {}, value: {} ".format(filename, CATEGORIES[index])
        		print(p)
