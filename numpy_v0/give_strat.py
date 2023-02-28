import numpy as np


def can_win(board, piece):
    # lines
    for i in range(4):
        if np.sum(board[i, :, 0], axis=0) == 3:
            s = np.sum(np.vstack((board[i, :, 1:], piece)), axis=0)
            if np.any(np.logical_or(s == 0, s == 4)):
                return True
    # columns
    for j in range(4):
        if np.sum(board[:, j, 0], axis=0) == 3:
            s = np.sum(np.vstack((board[:, j, 1:], piece)), axis=0)
            if np.any(np.logical_or(s == 0, s == 4)):
                return True
    # diagonal
    if np.sum(np.diag(board[:, :, 0])) == 3:
        s = np.sum(np.hstack((board[:, :, 1:].diagonal(
            0, 0, 1), piece[:, np.newaxis])), axis=1)
        if np.any(np.logical_or(s == 0, s == 4)):
            return True
    # diagonal bis
    if np.sum(np.diag(board[:, ::-1, 0])) == 3:
        s = np.sum(np.hstack((board[:, ::-1, 1:].diagonal(
            0, 0, 1), piece[:, np.newaxis])), axis=1)
        if np.any(np.logical_or(s == 0, s == 4)):
            return True
    # else can not win
    return False


def giveToNotLose(board, pieces):
    remaining_pieces_idx = np.nonzero(pieces[0, :] == 0)[0]
    non_winning_pieces_idx = []
    for idx in remaining_pieces_idx:
        if not can_win(board, pieces[1:, idx]):
            non_winning_pieces_idx.append(idx)
    # give non wining piece if possible
    if len(non_winning_pieces_idx) > 0:
        return np.random.choice(non_winning_pieces_idx)
    # else give any piece
    else:
        return np.random.choice(remaining_pieces_idx)
