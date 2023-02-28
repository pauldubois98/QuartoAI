import numpy as np
from game import *


def checkAlign_old(board):
    # lines
    l = np.logical_and(np.sum(board[:, :, 0], axis=0) == 4,
                       np.logical_or.reduce((np.sum(board[:, :, 1], axis=0) == 4,
                                             np.sum(
                                                 board[:, :, 1], axis=0) == 0,
                                             np.sum(
                                                 board[:, :, 2], axis=0) == 4,
                                             np.sum(
                                                 board[:, :, 2], axis=0) == 0,
                                             np.sum(
                                                 board[:, :, 3], axis=0) == 4,
                                             np.sum(
                                                 board[:, :, 3], axis=0) == 0,
                                             np.sum(
                                                 board[:, :, 4], axis=0) == 4,
                                             np.sum(board[:, :, 4], axis=0) == 0)))
    if np.any(l):
        return True
    # columns
    c = np.logical_and(np.sum(board[:, :, 0], axis=1) == 4,
                       np.logical_or.reduce((np.sum(board[:, :, 1], axis=1) == 4,
                                             np.sum(
                                                 board[:, :, 1], axis=1) == 0,
                                             np.sum(
                                                 board[:, :, 2], axis=1) == 4,
                                             np.sum(
                                                 board[:, :, 2], axis=1) == 0,
                                             np.sum(
                                                 board[:, :, 3], axis=1) == 4,
                                             np.sum(
                                                 board[:, :, 3], axis=1) == 0,
                                             np.sum(
                                                 board[:, :, 4], axis=1) == 4,
                                             np.sum(board[:, :, 4], axis=1) == 0)))
    if np.any(c):
        return True
    # diagonal
    d1 = np.logical_and(np.sum(np.diag(board[:, :, 0])) == 4,
                        np.logical_or.reduce((np.sum(np.diag(board[:, :, 1])) == 4,
                                              np.sum(
                                                  np.diag(board[:, :, 1])) == 0,
                                              np.sum(
                                                  np.diag(board[:, :, 2])) == 4,
                                              np.sum(
                                                  np.diag(board[:, :, 2])) == 0,
                                              np.sum(
                                                  np.diag(board[:, :, 3])) == 4,
                                              np.sum(
                                                  np.diag(board[:, :, 3])) == 0,
                                              np.sum(
                                                  np.diag(board[:, :, 4])) == 4,
                                              np.sum(np.diag(board[:, :, 4])) == 0)))
    if d1:
        return True
    # diagonal bis
    d2 = np.logical_and(np.sum(np.diag(board[:, ::-1, 0])) == 4,
                        np.logical_or.reduce((np.sum(np.diag(board[:, ::-1, 1])) == 4,
                                              np.sum(
                                             np.diag(board[:, ::-1, 1])) == 0,
                            np.sum(
                            np.diag(board[:, ::-1, 2])) == 4,
                            np.sum(
                            np.diag(board[:, ::-1, 2])) == 0,
                            np.sum(
                            np.diag(board[:, ::-1, 3])) == 4,
                            np.sum(
                            np.diag(board[:, ::-1, 3])) == 0,
                            np.sum(
                            np.diag(board[:, ::-1, 4])) == 4,
                            np.sum(np.diag(board[:, ::-1, 4])) == 0)))
    if d2:
        return True
    return False


def plotGame(stratPlaceA, stratGiveA, stratPlaceB, stratGiveB):
    global board, pieces
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
        plotPieces(pieces)
        plotBoard(board)
        if checkAlign(board):
            print('player A wins!')
            return 1
        piece_id = stratGiveA(board, pieces)

        pieces[0, piece_id] = 1
        piece = pieces[1:, piece_id]
        place_id_1, place_id_2 = stratPlaceB(board, pieces, piece)
        board[place_id_1, place_id_2, 0] = 1
        board[place_id_1, place_id_2, 1:] = piece
        plotPieces(pieces)
        plotBoard(board)
        if checkAlign(board):
            print('player B wins!')
            return -1

        piece_id = stratGiveB(board, pieces)

    pieces[0, piece_id] = 1
    piece = pieces[1:, piece_id]
    place_id_1, place_id_2 = stratPlaceA(board, pieces, piece)
    board[place_id_1, place_id_2, 0] = 1
    board[place_id_1, place_id_2, 1:] = piece
    plotPieces(pieces)
    plotBoard(board)
    if checkAlign(board):
        print('player A wins!')
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
    plotPieces(pieces)
    plotBoard(board)
    if checkAlign(board):
        print('player B wins!')
        return -1

    print('no winner!')
    return 0


# board = np.array([[[1, 1, 1, 1],
#                    [1, 1, 1, 1],
#                    [1, 1, 0, 1],
#                    [1, 1, 0, 1]],
#                   [[1, 1, 0, 0],
#                    [0, 0, 1, 0],
#                    [1, 1, 0, 0],
#                    [0, 1, 0, 1]],
#                   [[1, 1, 1, 0],
#                    [0, 0, 1, 1],
#                    [0, 1, 0, 1],
#                    [0, 0, 0, 0]],
#                   [[0, 0, 1, 0],
#                    [1, 0, 1, 1],
#                    [1, 1, 0, 0],
#                    [1, 0, 0, 1]],
#                   [[0, 1, 0, 1],
#                    [1, 0, 1, 1],
#                    [0, 0, 0, 0],
#                    [0, 1, 0, 1]]], dtype=np.uint8).swapaxes(0, 2)
# print(checkAlign(board))
# print(checkAlign2(board))

# plotBoard(board)
