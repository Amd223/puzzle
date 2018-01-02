import os
import cv2
from PIL import Image

from puzzle.tools.utils import input_image, img_read
from puzzle.tools.crop import crop_interactive


img_path = input_image("Input an image name from '{}': \n")
print('Using image src : ' + img_path)

crop_path, _ = crop_interactive(img_path)
print('Using crop_path : ' + crop_path)

# Load images
img1 = img_read(crop_path) # queryImage
img2 = img_read(img_path)  # trainImage
print('crop', crop_path, img1)
print('train', img_path, img2)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the key points and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = [[m] for m, n in matches if m.distance < 0.75*n.distance]

# cv2.drawMatchesKnn expects list of lists as matches.
img3_path = os.path.join(os.path.dirname(crop_path), 'sift_key_points.jpg')
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imwrite(img3_path, img3)
print('img3', img3_path)


Image.open(img3_path).show()