import torch
from game import checkAlign
from random_player import randomPlace, randomGive


def checkAlmostAlign(board):
    """check if there are 3 pieces of the same shape aligned"""
    # columns
    columns_sums = board.sum(dim=1)
    columns_full_pieces = columns_sums[0, :] == 3
    columns_same_shapes = torch.logical_or(
        columns_sums[1:, :] == 0, columns_sums[1:, :] == 3).any(dim=1)
    if torch.logical_and(columns_full_pieces, columns_same_shapes).any():
        return "c"
    # lines
    lines_sums = board.sum(dim=2)
    lines_full_pieces = lines_sums[0, :] == 3
    lines_same_shapes = torch.logical_or(
        lines_sums[1:, :] == 0, lines_sums[1:, :] == 3).any(dim=1)
    if torch.logical_and(lines_full_pieces, lines_same_shapes).any():
        return "l"
    # diagonal
    diagonal_sums = board.diagonal(0, 1, 2).sum(dim=1)
    diagonal_full_pieces = diagonal_sums[0] == 3
    diagonal_same_shapes = torch.logical_or(
        diagonal_sums[1:] == 0, diagonal_sums[1:] == 3).any()
    if torch.logical_and(diagonal_full_pieces, diagonal_same_shapes).any():
        return "d"
    # diagonal bis
    flipped_board = torch.flip(board, (1,))
    diagonal_bis_sums = flipped_board.diagonal(0, 1, 2).sum(dim=1)
    diagonal_bis_full_pieces = diagonal_bis_sums[0] == 3
    diagonal_bis_same_shapes = torch.logical_or(
        diagonal_sums[1:] == 0, diagonal_sums[1:] == 3).any()
    if torch.logical_and(diagonal_bis_full_pieces, diagonal_bis_same_shapes).any():
        return "D"
    # no align
    return False


def heuristicPlace(board, piece):
    """try to place the piece in the first winning available place"""
    # if not 3 in a row, return None
    if not checkAlmostAlign(board):
        return None
    # if we can, place the piece in the first winning available place
    available_places = torch.where(board[0, :, :] == 0)
    for indice in range(available_places[0].shape[0]):
        test_place_i = available_places[0][indice]
        test_place_j = available_places[1][indice]
        test_board = board.clone()
        test_board[0, test_place_i, test_place_j] = 1
        test_board[1:, test_place_i, test_place_j] = piece
        if checkAlign(test_board):
            return test_place_i, test_place_j
    # else, return None
    return None


def heuristicGive(board, pieces):
    """avoid giving the piece that will make the opponent win"""
    available_pieces = torch.where(pieces[0, :] == 1)
    for test_place in available_pieces[0]:
        test_piece = pieces[1:, test_place]
        if heuristicPlace(board, test_piece) is not None:
            return test_place
    return None


class HeuristicPlayer:
    def give(self, board, pieces):
        pos = heuristicGive(board, pieces)
        if pos is None:
            return randomGive(pieces)
        return pos

    def place(self, board, pieces, piece):
        pos = heuristicPlace(board, piece)
        if pos is None:
            return randomPlace(board)
        return pos
