import glob
import os

from puzzle.tools.image_manip import img_resize
from puzzle.tools.utils import input_directory


def resize_all(dir_in, **kwargs):
    dir_out = os.path.join(dir_in, "resized")
    os.makedirs(dir_out, exist_ok=True)

    images = glob.glob(os.path.join(dir_in, "*.*"))
    for img_path_in in images:
        rel_path = os.path.relpath(img_path_in, dir_in)
        img_path_out = os.path.join(dir_out, rel_path)
        img_resize(img_path_in, img_path_out, **kwargs)


if __name__ == "__main__":
    dir_in = input_directory()
    resize_all(dir_in)