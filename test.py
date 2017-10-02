import cv2 as opencv
import sklearn
import numpy
from sklearn.feature_extraction import image
from PIL import Image

im = Image.open("adam.jpg")

one_image = numpy.arange(16).reshape((4, 4))  # create an image as a 2-dimensional array of 16 pixels

#  print(one_image)

patches = image.extract_patches_2d(im, (2, 2), 4) #b reaks up the image into 4 2-dimensional arrays of 4
print(patches)
