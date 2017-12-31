import PIL
import cv2 as opencv
import sklearn
import matplotlib
import numpy
from sklearn.feature_extraction import image
from skimage import io

#print(io.available_plugins)
io.use_plugin('pil')  # sets the skimage library to use PIL to read and write images

im = io.imread("adam.jpg")
patches = image.extract_patches_2d(im, (486, 486), 4)  # breaks up the image into 4 2-dimensional arrays of 4
io.imsave("adampatch.jpg", patches[0])  # saves the patch 1 out of 4



# one_image = numpy.arange(16).reshape((4, 4))  # create an image as a 2-dimensional array of 16 pixels