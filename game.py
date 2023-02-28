import torch
import numpy as np
import matplotlib.pyplot as plt


def plotPieces(pieces):
    for i in range(pieces.shape[1]):
        if (pieces[0, i] == 0):
            print(f'*{i:x}*', end=' ')
        else:
            print(f' {i:x} ', end=' ')
    print()


def plotBoard(board):
    im = (board[1, :, :]+(2*board[2, :, :])+(4*board[3, :, :]) +
          (8*board[4, :, :]))
    for i in range(4):
        for j in range(4):
            if board[0, i, j] == 0:
                print(' ', end='')
            else:
                print(f'{im[i,j]:x}', end='')
        print()
    print()


def init_game():
    board = torch.zeros((5, 4, 4), dtype=torch.int8)
    pieces = torch.zeros((5, 16), dtype=torch.int8)
    pieces[0, :] = 1
    pieces[1, 1::2] = 1
    pieces[2, 2::4] = 1
    pieces[2, 3::4] = 1
    pieces[3, 4:8] = 1
    pieces[3, 12:] = 1
    pieces[4, 8:] = 1
    return board, pieces


def checkAlign(board):
    # columns
    columns_sums = board.sum(dim=1)
    columns_full_pieces = columns_sums[0, :] == 4
    columns_same_shapes = torch.logical_or(
        columns_sums[1:, :] == 0, columns_sums[1:, :] == 4).any(dim=1)
    if torch.logical_and(columns_full_pieces, columns_same_shapes).any():
        return "c"
    # lines
    lines_sums = board.sum(dim=2)
    lines_full_pieces = lines_sums[0, :] == 4
    lines_same_shapes = torch.logical_or(
        lines_sums[1:, :] == 0, lines_sums[1:, :] == 4).any(dim=1)
    if torch.logical_and(lines_full_pieces, lines_same_shapes).any():
        return "l"
    # diagonal
    diagonal_sums = board.diagonal(0, 1, 2).sum(dim=1)
    diagonal_full_pieces = diagonal_sums[0] == 4
    diagonal_same_shapes = torch.logical_or(
        diagonal_sums[1:] == 0, diagonal_sums[1:] == 4).any()
    if torch.logical_and(diagonal_full_pieces, diagonal_same_shapes).any():
        return "d"
    # diagonal bis
    flipped_board = torch.flip(board, (1,))
    diagonal_bis_sums = flipped_board.diagonal(0, 1, 2).sum(dim=1)
    diagonal_bis_full_pieces = diagonal_bis_sums[0] == 4
    diagonal_bis_same_shapes = torch.logical_or(
        diagonal_sums[1:] == 0, diagonal_sums[1:] == 4).any()
    if torch.logical_and(diagonal_bis_full_pieces, diagonal_bis_same_shapes).any():
        return "D"
    # no align
    return False


def full_game(playerA, playerB):
    global board, pieces
    board, pieces = init_game()
    piece_idx = playerB.give(board, pieces)
    for turn in range(16):
        if turn % 2 == 0:
            piece = pieces[1:, piece_idx]
            place_i, place_j = playerA.place(board, pieces, piece)
            board[0, place_i, place_j] = 1
            board[1:, place_i, place_j] = piece
            pieces[0, piece_idx] = 0
            if checkAlign(board):
                return "A"
            piece_idx = playerA.give(board, pieces)
        else:
            piece = pieces[1:, piece_idx]
            place_i, place_j = playerB.place(board, pieces, piece)
            board[0, place_i, place_j] = 1
            board[1:, place_i, place_j] = piece
            pieces[0, piece_idx] = 0
            if checkAlign(board):
                return "B"
            if turn == 15:
                return "Draw"
            piece_idx = playerB.give(board, pieces)
        # print('   turn', turn)
        # plotBoard(board)
        # plotPieces(pieces)
