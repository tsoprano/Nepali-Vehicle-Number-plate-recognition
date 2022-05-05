import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0
y = np.array(y)

dense_layers = [0, 1, 2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print ("NAME============> ",NAME)
            tensorBoard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for i in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for i in range(dense_layer):
                if i>0:
                    model.add(Dense(layer_size))
                    model.add(Activation("relu"))

            model.add(Dense(10))
            model.add(Activation("sigmoid"))

            model.compile(loss = 'sparse_categorical_crossentropy',
                         optimizer = 'adam',
                         metrics = ['accuracy'])

            model.fit(X, y, batch_size = 20, epochs=20, validation_split = 0.1, callbacks = [tensorBoard])

            #model.save("64x2-CNN.model")