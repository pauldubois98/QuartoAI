import torch
import random
from game import checkAlign
from random_player import randomPlace, randomGive


def calculate_tension(board):
    """compute the tension of the board"""
    # columns
    columns_sums = board.sum(dim=1)
    columns_full_pieces = columns_sums[0, :] == 3
    columns_same_shapes = torch.logical_or(
        columns_sums[1:, :] == 0, columns_sums[1:, :] == 3).any(dim=1)
    columns_tension = torch.logical_and(
        columns_full_pieces, columns_same_shapes)
    # lines
    lines_sums = board.sum(dim=2)
    lines_full_pieces = lines_sums[0, :] == 3
    lines_same_shapes = torch.logical_or(
        lines_sums[1:, :] == 0, lines_sums[1:, :] == 3).any(dim=1)
    lines_tension = torch.logical_and(lines_full_pieces, lines_same_shapes)
    # diagonal
    diagonal_sums = board.diagonal(0, 1, 2).sum(dim=1)
    diagonal_full_pieces = diagonal_sums[0] == 3
    diagonal_same_shapes = torch.logical_or(
        diagonal_sums[1:] == 0, diagonal_sums[1:] == 3).any()
    diagonal_tension = torch.logical_and(
        diagonal_full_pieces, diagonal_same_shapes)
    # diagonal bis
    flipped_board = torch.flip(board, (1,))
    diagonal_bis_sums = flipped_board.diagonal(0, 1, 2).sum(dim=1)
    diagonal_bis_full_pieces = diagonal_bis_sums[0] == 3
    diagonal_bis_same_shapes = torch.logical_or(
        diagonal_sums[1:] == 0, diagonal_sums[1:] == 3).any()
    diagonal_bis_tension = torch.logical_and(
        diagonal_bis_full_pieces, diagonal_bis_same_shapes)
    # tension
    total_tension = lines_tension.sum().item() \
        + columns_tension.sum().item() \
        + diagonal_tension.sum().item() \
        + diagonal_bis_tension.sum().item()
    return total_tension


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


def maximizeTensionPlace(board, piece):
    """maximize the tension of the board"""
    available_places = torch.where(board[0, :, :] == 0)
    max_tension = 0
    max_place = None
    max_nb = 0
    for indice in range(available_places[0].shape[0]):
        test_place_i = available_places[0][indice]
        test_place_j = available_places[1][indice]
        test_board = board.clone()
        test_board[0, test_place_i, test_place_j] = 1
        test_board[1:, test_place_i, test_place_j] = piece
        tension = calculate_tension(test_board)
        if tension > max_tension:
            max_tension = tension
            max_place = (test_place_i, test_place_j)
            max_nb = 1
        elif tension == max_tension:
            if random.randint(0, max_nb) == 0:
                max_place = (test_place_i, test_place_j)
            max_nb += 1
    return max_place


def minimizeTensionPlace(board, piece):
    """minimize the tension of the board"""
    available_places = torch.where(board[0, :, :] == 0)
    min_tension = 100
    min_place = None
    min_nb = 0
    for indice in range(available_places[0].shape[0]):
        test_place_i = available_places[0][indice]
        test_place_j = available_places[1][indice]
        test_board = board.clone()
        test_board[0, test_place_i, test_place_j] = 1
        test_board[1:, test_place_i, test_place_j] = piece
        tension = calculate_tension(test_board)
        if tension < min_tension:
            min_tension = tension
            min_place = (test_place_i, test_place_j)
            min_nb = 1
        elif tension == min_tension:
            if random.randint(0, min_nb) == 0:
                min_place = (test_place_i, test_place_j)
            min_nb += 1
    return min_place


class HeuristicPlayer:
    """player that tries to win when placing & avoid losing when giving"""

    def __str__(self) -> str:
        return "Heuristic Player"
    
    def give(self, board, pieces):
        pos = heuristicGive(board, pieces)
        if pos is not None:
            return pos
        return randomGive(pieces)

    def place(self, board, pieces, piece):
        pos = heuristicPlace(board, piece)
        if pos is not None:
            return pos
        return randomPlace(board)


class AggressiveHeuristicPlayer:
    """player that tries to win when placing & avoid losing when giving, playing aggressively"""

    def __str__(self) -> str:
        return "Aggressive Heuristic Player"
    
    def give(self, board, pieces):
        pos = heuristicGive(board, pieces)
        if pos is not None:
            return pos
        return randomGive(pieces)

    def place(self, board, pieces, piece):
        pos = heuristicPlace(board, piece)
        if pos is not None:
            return pos
        pos = maximizeTensionPlace(board, piece)
        if pos is not None:
            return pos
        return randomPlace(board)


class PacificHeuristicPlayer:
    """player that tries to win when placing & avoid losing when giving, playing pacifically"""

    def __str__(self) -> str:
        return "Pacific Heuristic Player"
    
    def give(self, board, pieces):
        pos = heuristicGive(board, pieces)
        if pos is not None:
            return pos
        return randomGive(pieces)

    def place(self, board, pieces, piece):
        pos = heuristicPlace(board, piece)
        if pos is not None:
            return pos
        pos = minimizeTensionPlace(board, piece)
        if pos is not None:
            return pos
        return randomPlace(board)


