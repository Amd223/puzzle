from time import time

from keras.applications import VGG16
from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.callbacks import TensorBoard
import numpy as np
import cv2
import os

ROOT_PATH = "/Users/AliciaDaurignac/Documents/Exeter/3eme_annee/Dissertation/Project/images"
train_set_paths_class1 = os.path.join(ROOT_PATH, "art/resized")
train_set_paths_class2 = os.path.join(ROOT_PATH, "merged")

test_set_paths_class1 = os.path.join(ROOT_PATH, "test_set")
test_set_paths_class2 = os.path.join(ROOT_PATH, "merged_set")


batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 200 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 16 # we will use 2x2 pooling throughout (the bigger this number the smaller the image)
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

X_train = []
y_train = []

for file in os.listdir(train_set_paths_class1):
    if not file.endswith(".jpg"):
        continue
    X_train.append(cv2.imread(os.path.join(train_set_paths_class1, file)))
    y_train.append([0])

for file in os.listdir(train_set_paths_class2):
    if not file.endswith(".jpg"):
        continue
    X_train.append(cv2.imread(os.path.join(train_set_paths_class2, file)))
    y_train.append([1])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = X_train
y_test = y_train

# num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10
# num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = np.unique(y_train).shape[0] # there are 10 image classes

# X_train = X_train.astype('float32') #turning images to arrays of floats
# X_test = X_test.astype('float32')
# X_train /= np.max(X_train) # Normalise data to [0, 1] range
# X_test /= np.max(X_test) # Normalise data to [0, 1] range
#
# # telling the ground truch of probability distributions used to train
# Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
# Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)


# # Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
# conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
# conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
# pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
# drop_1 = Dropout(drop_prob_1)(pool_1)
# # Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
# conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
# conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
# pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
# drop_2 = Dropout(drop_prob_1)(pool_2)
# # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax

model = VGG16(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

last_layer = model.get_layer('pool5').output

# don't touch!
# flat = Flatten()(last_layer)
hidden = Dense(hidden_size, activation='relu')(last_layer) # you need it
drop_3 = Dropout(last_layer)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

model.fit(X_train, Y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs, steps_per_epoch= 72,
          verbose=1, validation_split=0.1, callbacks=[tensorboard]) # ...holding out 10% of the data for validation
