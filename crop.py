import os
import cv2

curr_dir   = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(curr_dir, 'w')
os.makedirs(target_dir, exist_ok=True)   # mkdir -p

width_blocks  = 486
height_blocks = 486

img = cv2.imread("adam.jpg")
height, width = img.shape[:2]

# Iterate in the range(begin, end, step)
for y in range(0, height, height_blocks):
    for x in range(0, width, width_blocks):
        crop_name = os.path.join(target_dir, 'img_y-%d_x-%d.png' % (y, x))
        crop_img  = img[y:y+height_blocks, x:x+width_blocks]
        print('Creating %s...' % crop_name)
        cv2.imwrite(crop_name, crop_img)
