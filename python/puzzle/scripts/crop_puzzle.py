import os

from puzzle.scripts.get_all_images import crop_one
from puzzle.tools.utils import input_image, input_directory



if __name__ == "__main__":
    img_path = input_directory()
    resized_dir = os.path.join("../../../", "resized")
    crop_one(img_path, (48, 48), crop_pos=(0, 0), save=True)


    #print("Crops dropped in '{}'".format(dir_out))