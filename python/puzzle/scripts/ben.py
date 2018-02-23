import tensorboard
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate
from keras import backend as K
import numpy as np
import os

from puzzle.scripts.get_all_images import create_training_set

def get_activation_maps():

    ROOT_PATH = "/Users/AliciaDaurignac/Documents/Exeter/3eme_annee/Dissertation/Project/images"
    train_set_paths_class1 = os.path.join(ROOT_PATH, "art/resized")
    train_set_paths_class2 = os.path.join(ROOT_PATH, "merged")

    test_set_paths_class1 = os.path.join(ROOT_PATH, "test_set")
    test_set_paths_class2 = os.path.join(ROOT_PATH, "merged_set")

    batch_size = 32 # in each iteration, we consider 32 training examples at once
    num_epochs = 200 # we iterate 200 times over the entire training set

    hidden_size = 512 # the FC layer will have 512 neurons

    X_train1, X_train2, y_train = create_training_set()
    X_train= (X_train1, X_train2)

    # for file in os.listdir(train_set_paths_class1):
    #     if not file.endswith(".jpg"):
    #         continue
    #     img = image.load_img(os.path.join(train_set_paths_class1, file), target_size=(224,224))
    #     x = image.img_to_array(img)
    #     X_train.append(x)
    #     y_train.append([0])
    #
    # for file in os.listdir(train_set_paths_class2):
    #     if not file.endswith(".jpg"):
    #         continue
    #     X_train.append(cv2.imread(os.path.join(train_set_paths_class2, file)))
    #     y_train.append([1])

    X_train = preprocess_input(np.array(X_train))
    y_train = preprocess_input(np.array(y_train))

    X_test = X_train
    y_test = y_train

    # create the base pre-trained models
    base_model = VGG16(weights='imagenet', include_top=False)
    base_model2 = VGG16(weights='imagenet', include_top=False)


    # add a global spatial average pooling layer
    x = base_model.output
    x2 = base_model2.output

    # concat x1 and x2
    concatenated = Concatenate(axis=-1)(x, x2)

    return concatenated

# let's add a fully-connected layer
hidden_layer = Dense(1024, activation='relu')(concatenated)
# and a logistic layer -- let's say we have 2 classes
predictions = Dense(2, activation='softmax')(hidden_layer)

# this is the model we will train
model = Model(inputs=[base_model.input, base_model2.input], outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model2.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit(([X_train1, X_train2]), y_train,                # Train the model using the training set...
          batch_size=batch_size, epochs=num_epochs, steps_per_epoch= 72,
          verbose=1, validation_split=0.1, callbacks=[tensorboard]) # ...holding out 10% of the data for validation

#model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!