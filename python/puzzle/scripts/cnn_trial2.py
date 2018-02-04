from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


model = VGG16(weights='imagenet', include_top=False)

img_path = 'images/adam.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

img_path2 = 'images/Lenna.jpg'
img2 = image.load_img(img_path2, target_size=(224, 224))
y = image.img_to_array(img2)
y = np.expand_dims(y, axis=0)
y = preprocess_input(y)

features = model.predict(x)
features2 = model.predict(y)

print("FEATURES 1:", features)
print("FEATURES 2:", features2)