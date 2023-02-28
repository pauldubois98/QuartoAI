import numpy as np


def placeRandom(board, pieces, piece):
    remaining_places_idx = np.nonzero(board[:, :, 0].flatten() == 0)
    place_idx = np.random.choice(remaining_places_idx[0])
    place_double_idx = (place_idx//4, place_idx % 4)
    return place_double_idx


def giveRandom(board, pieces):
    remaining_pieces_idx = np.nonzero(pieces[0, :] == 0)
    piece_idx = np.random.choice(remaining_pieces_idx[0])
    return piece_idx
