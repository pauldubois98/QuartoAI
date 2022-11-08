import numpy as np
import random
from random_strat import placeRandom


def placeToWin(board, pieces, piece, else_strat=placeRandom):
    # lines
    for i in range(4):
        if np.sum(board[i, :, 0], axis=0) == 3:
            s = np.sum(np.vstack((board[i, :, 1:], piece)), axis=0)
            if np.any(np.logical_or(s == 0, s == 4)):
                j = np.where(board[i, :, 0] == 0)[0][0]
                return i, j
    # columns
    for j in range(4):
        if np.sum(board[:, j, 0], axis=0) == 3:
            s = np.sum(np.vstack((board[:, j, 1:], piece)), axis=0)
            if np.any(np.logical_or(s == 0, s == 4)):
                i = np.where(board[:, j, 0] == 0)[0][0]
                return i, j
    # diagonal
    if np.sum(np.diag(board[:, :, 0])) == 3:
        s = np.sum(np.hstack((board[:, :, 1:].diagonal(
            0, 0, 1), piece[:, np.newaxis])), axis=1)
        if np.any(np.logical_or(s == 0, s == 4)):
            i = np.where(np.diag(board[:, :, 0]) == 0)[0][0]
            return i, i
    # diagonal bis
    if np.sum(np.diag(board[:, ::-1, 0])) == 3:
        s = np.sum(np.hstack((board[:, ::-1, 1:].diagonal(
            0, 0, 1), piece[:, np.newaxis])), axis=1)
        if np.any(np.logical_or(s == 0, s == 4)):
            j = np.where(np.diag(board[:, ::-1, 0]) == 0)[0][0]
            return j, 3-j
    # else strat
    return else_strat(board, pieces, piece)


def tension(board):
    t = 0
    # lines
    l = np.logical_and(np.sum(board[:, :, 0], axis=0) == 3,
                       np.any(np.logical_or(np.sum(board[:, :, 1:], axis=0) == 3,
                                            np.sum(board[:, :, 1:], axis=0) == 0), axis=1))
    t += np.sum(l)
    # columns
    c = np.logical_and(np.sum(board[:, :, 0], axis=1) == 4,
                       np.any(np.logical_or(np.sum(board[:, :, 1:], axis=1) == 4,
                                            np.sum(board[:, :, 1:], axis=1) == 0), axis=1))
    t += np.sum(c)
    # diagonal
    d1 = np.logical_and(np.sum(np.diag(board[:, :, 0])) == 4,
                        np.any(np.logical_or(np.sum(board[:, :, 1:].diagonal(0, 0, 1), axis=1) == 4,
                                             np.sum(board[:, :, 1:].diagonal(0, 0, 1), axis=1) == 0)))
    t += np.sum(d1)
    # diagonal bis
    d2 = np.logical_and(np.sum(np.diag(board[:, ::-1, 0])) == 4,
                        np.any(np.logical_or(np.sum(board[:, ::-1, 1:].diagonal(0, 0, 1), axis=1) == 4,
                                             np.sum(board[:, ::-1, 1:].diagonal(0, 0, 1), axis=1) == 0)))
    t += np.sum(d2)
    return t


def maxTension(board, pieces, piece):
    remaining_places_idx = np.nonzero(board[:, :, 0] == 0)
    choices = []
    for i, j in zip(*remaining_places_idx):
        temp_board = np.copy(board)
        temp_board[i, j, 0] = 1
        temp_board[i, j, 1:] = piece
        t = tension(temp_board)
        choices.append((t, (i, j)))
    random.shuffle(choices)
    choices.sort(reverse=True)
    return choices[0][1]


def minTension(board, pieces, piece):
    remaining_places_idx = np.nonzero(board[:, :, 0] == 0)
    choices = []
    for i, j in zip(*remaining_places_idx):
        temp_board = np.copy(board)
        temp_board[i, j, 0] = 1
        temp_board[i, j, 1:] = piece
        t = tension(temp_board)
        choices.append((t, (i, j)))
    random.shuffle(choices)
    choices.sort()
    return choices[0][1]


def placeToWin_random(board, pieces, piece):
    return placeToWin(board, pieces, piece, placeRandom)


def placeToWin_minTension(board, pieces, piece):
    return placeToWin(board, pieces, piece, minTension)


def placeToWin_maxTension(board, pieces, piece):
    return placeToWin(board, pieces, piece, maxTension)
