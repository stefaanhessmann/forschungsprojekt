# 3. Import libraries and modules
import numpy as np

np.random.seed(123)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import keras
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train = X_train
Y_train = Y_train
# 7. Define model architecture
load_model = True
if load_model:
    model=keras.models.load_model("MyModel.h5")
else:
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # 8. Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

# 9. Fit model on training data
learn = True
if learn:
    model.fit(X_train, Y_train,
              batch_size=32, nb_epoch=1, verbose=1)
    model.save("MyModel.h5")
# 10. Evaluate model on test data
#score = model.evaluate(X_test, Y_test, verbose=0)

"""
img = mpimg.imread('myNumbers/8.png')
returns = model.predict(X_test)
labels = [np.argmax(i) for i in returns]
for i in range(1):
    f = plt.figure()
    plt.imshow(X_test[i].reshape(28, 28))
    plt.xlabel("{}".format(labels[i]))
    plt.savefig("./{}/result{}".format(labels[i], i))
"""

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

for i, file in enumerate(['a', 'b', 'c', 'd']):
    img = mpimg.imread('myNumbers/{}.png'.format(file))
    gray = 1-rgb2gray(img)
    gray = gray.reshape(1, 28, 28, 1)
    returns = model.predict(gray)
    labels = [np.argmax(i) for i in returns]
    f = plt.figure()
    plt.imshow(gray.reshape(28, 28))
    plt.xlabel("Number = {}".format(labels[0]))
    plt.savefig("./myNumbers/result{}is{}".format(file, labels[0]))
