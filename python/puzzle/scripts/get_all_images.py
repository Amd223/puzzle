import os
import glob

curr_dir = os.path.dirname(__file__)
img_dir = os.path.realpath(os.path.join(curr_dir, '../../../images'))

for image in glob.iglob(img_dir + '/**/*.jpg', recursive=True):
    print(image)
