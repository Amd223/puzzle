import cv2
from matplotlib import pyplot as plt

from puzzle.tools.utils import input_image, img_read
from puzzle.tools.crop import crop_interactive


img_path = input_image("Input an image name from '{}': \n")
print('Using image src : ' + img_path)

# crop_path, _ = crop_interactive(img_path, show_crop=False)
crop_path = input_image("Input a query image name from '{}': \n")

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
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

plt.imshow(img3), plt.show()


# apply template matching using SIFT - xfeatures does not work
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
# image = cv2.drawKeypoints(gray, kp, image)
# cv2.imwrite('sift_keypoints.jpg', image)