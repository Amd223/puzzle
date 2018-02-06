import os

from puzzle.tools.crop import crop_loulou, crop
from puzzle.tools.utils import input_image, input_directory



if __name__ == "__main__":
    dir_in = input_directory()
    resized_dir = os.path.join(dir_in, "resized")
    img = input_image(images_relpath=resized_dir)
    img_path = os.path.join(resized_dir, img )
    crop(img_path, (512,512))


    #print("Crops dropped in '{}'".format(dir_out))