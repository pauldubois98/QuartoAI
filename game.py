from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from game import *
from random_strat import *
from give_strat import *
from place_strat import *


def plotPieces(pieces):
    for i in range(pieces.shape[1]):
        if(pieces[0, i] != 0):
            print(f'*{i}*', end=' ')
        else:
            print(f' {i} ', end=' ')
    print()


def plotBoard(board):
    im = (board[:, :, 1]+(2*board[:, :, 2])+(4*board[:, :, 3]) +
          (8*board[:, :, 4])) + (99 * (1-board[:, :, 0]))
    for i in range(4):
        print('|', end='')
        for j in range(4):
            if board[i, j, 0] == 0:
                print('  ', end='|')
            else:
                print(f'{im[i,j]:02d}', end='|')
        print()
    print()
    # plt.imshow()


def checkAlign(board):
    # lines
    l = np.logical_and(np.sum(board[:, :, 0], axis=0) == 4,
                       np.any(np.logical_or(np.sum(board[:, :, 1:], axis=0) == 4,
                                            np.sum(board[:, :, 1:], axis=0) == 0), axis=1))
    if np.any(l):
        return True
    # columns
    c = np.logical_and(np.sum(board[:, :, 0], axis=1) == 4,
                       np.any(np.logical_or(np.sum(board[:, :, 1:], axis=1) == 4,
                                            np.sum(board[:, :, 1:], axis=1) == 0), axis=1))
    if np.any(c):
        return True
    # diagonal
    d1 = np.logical_and(np.sum(np.diag(board[:, :, 0])) == 4,
                        np.any(np.logical_or(np.sum(board[:, :, 1:].diagonal(0, 0, 1), axis=1) == 4,
                                             np.sum(board[:, :, 1:].diagonal(0, 0, 1), axis=1) == 0)))
    if np.any(d1):
        return True
    # diagonal bis
    d2 = np.logical_and(np.sum(np.diag(board[:, ::-1, 0])) == 4,
                        np.any(np.logical_or(np.sum(board[:, ::-1, 1:].diagonal(0, 0, 1), axis=1) == 4,
                                             np.sum(board[:, ::-1, 1:].diagonal(0, 0, 1), axis=1) == 0)))
    if np.any(d2):
        return True
    return False


def game(stratPlaceA, stratGiveA, stratPlaceB, stratGiveB):
    board = np.zeros((4, 4, 5), dtype=np.uint8)
    pieces = np.zeros((5, 16), dtype=np.uint8)
    pieces[1, 1::2] = 1
    pieces[2, 2::4] = 1
    pieces[2, 3::4] = 1
    pieces[3, 4:8] = 1
    pieces[3, 12:] = 1
    pieces[4, 8:] = 1

    piece_id = 0
    for i in range(7):
        pieces[0, piece_id] = 1
        piece = pieces[1:, piece_id]
        place_id_1, place_id_2 = stratPlaceA(board, pieces, piece)
        board[place_id_1, place_id_2, 0] = 1
        board[place_id_1, place_id_2, 1:] = piece
        if checkAlign(board):
            # player A wins
            return 1
        piece_id = stratGiveA(board, pieces)

        pieces[0, piece_id] = 1
        piece = pieces[1:, piece_id]
        place_id_1, place_id_2 = stratPlaceB(board, pieces, piece)
        board[place_id_1, place_id_2, 0] = 1
        board[place_id_1, place_id_2, 1:] = piece
        if checkAlign(board):
            # player B wins
            return -1

        piece_id = stratGiveB(board, pieces)

    pieces[0, piece_id] = 1
    piece = pieces[1:, piece_id]
    place_id_1, place_id_2 = stratPlaceA(board, pieces, piece)
    board[place_id_1, place_id_2, 0] = 1
    board[place_id_1, place_id_2, 1:] = piece
    if checkAlign(board):
        # player A wins
        return 1

    remaining_pieces_idx = np.nonzero(pieces[0, :] == 0)
    piece_id = remaining_pieces_idx[0][0]
    pieces[0, piece_id] = 1
    piece = pieces[1:, piece_id]
    remaining_places_idx = np.nonzero(board[:, :, 0].flatten() == 0)
    place_idx = remaining_places_idx[0][0]
    place_id_1, place_id_2 = place_idx//4, place_idx % 4
    board[place_id_1, place_id_2, 0] = 1
    board[place_id_1, place_id_2, 1:] = piece
    if checkAlign(board):
        # player B wins
        return -1

    # no winner
    return 0


def full_test(stratPlaceA, stratGiveA, stratPlaceB, stratGiveB, N=10**3):
    res = np.zeros(N, dtype=np.int32)
    for i in range(N):
        res[i] = game(stratPlaceA, stratGiveA, stratPlaceB, stratGiveB)
    f = np.sum(res == 1)/N
    print(f'wins: {100*f:.2f}±{100*1.96*np.sqrt(f*(1-f))/np.sqrt(N):.2f}%')
    f = np.sum(res == 0)/N
    print(f'ties: {100*f:.2f}±{100*1.96*np.sqrt(f*(1-f))/np.sqrt(N):.2f}%')
    f = np.sum(res == -1)/N
    print(f'lost: {100*f:.2f}±{100*1.96*np.sqrt(f*(1-f))/np.sqrt(N):.2f}%')

    s = np.sum(res)
    print(f'overall: {s/N:+.3f}')

    plt.plot(np.cumsum(res))
    plt.show()


if __name__ == '__main__':
    game(placeRandom, giveRandom, placeRandom, giveRandom)
