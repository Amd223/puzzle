import numpy as np
import os

from puzzle.scripts.merge_all_dir import merge
from puzzle.scripts.reconstruction import crop_puzzle
from puzzle.tools.utils import input_image, input_directory, img_read
from puzzle.tools.crop import crop

def find_minimum(list):

    n = 10000000
    for i in list:
        if i ==0:
            pass
        elif i <= n:
            n = i
    position = list.index(n)
    return n, position

def find_next_lowest_l2(img_piece, list):
    """
    Given an image piece and a list, return the piece with the lowest l2 distance
    :param img_piece:
    :param list:
    :return:
    """

    distances = []

    for i in list:
        distance = l2_distance_pieces(img_piece, i)
        distances.append(distance)
    lowest, index = find_minimum(distances)
    next_piece = list[index]
    list.pop(index)

    return next_piece, index

def l2_distance_image(img_path):
    """
    Calculates the l2 distance between all combinations of pieces of the puzzle
    :param img_path:
    :return: array with puzzle pieces correctly ordered
    """
    # distances = []
    # final_distances = []
    # piece = []
    # used = []
    #
    # for i in range(len(pieces)):
    #     for j in range(len(pieces)):
    #         distance = l2_distance_pieces(pieces[i], pieces[j])
    #         distances.append(distance)
    #
    #     low, position = find_minimum(distances)
    #     if i == position:
    #         break
    #     if position in used:
    #         break
    #         piece.append((i,position))
    #         final_distances.append(low)
    #         distances = []
    #         used.append(position)
    #

    pieces = crop_puzzle(img_path)
    ordered_pieces = []
    starting_piece = pieces[0]
    ordered_pieces.append(starting_piece)

   # while len(pieces) > 0:
    for i in range(10): # there are 11 puzzle pieces horizontally
        next_piece, index = find_next_lowest_l2(starting_piece, pieces)
        ordered_pieces.append(next_piece)
        starting_piece = next_piece
        pieces.pop(index)
        pieces.pop()

        # add a function to find the piece that goes below pieces[0]
        # extend it to use only last strip of images and see if there is more accuracy

    return ordered_pieces


def l2_distance_pieces(img1, img2):
    """
    :param img: original image
    :param template: cropped section of the image
    :return: l2 distance between images
    """

    distances = np.sqrt(np.sum(np.square(img1 - img2), axis=1))[:,None,None]
    return distances.sum()

if __name__ == "__main__":

    img_path = input_directory()
    distances = l2_distance_image(img_path)
    print(len(distances))



