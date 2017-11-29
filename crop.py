import cv2

width_blocks = 5
height_blocks = 5

img = cv2.imread("adam.jpg")
height, width = img.shape[:2]

block_height = height/height_blocks
block_width = width/width_blocks


for row in range(height_blocks):
    for column in range(width_blocks):
        start_x = column * block_width
        start_y = row * block_height
        crop = img[start_x, start_y, block_width, block_height]