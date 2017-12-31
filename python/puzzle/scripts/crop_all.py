from puzzle.tools.crop import crop_all
from puzzle.tools.utils import input_image


img_path = input_image()

dir_out = crop_all(img_path, (486, 486))
print("Crops dropped in '{}'".format(dir_out))