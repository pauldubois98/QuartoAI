import torch
from game import checkAlign
from random_player import randomPlace, randomGive


def heuristicPlace(board, piece):
    available_places = torch.where(board[0, :, :] == 0)
    for indice in range(available_places[0].shape[0]):
        test_place_i = available_places[0][indice]
        test_place_j = available_places[1][indice]
        test_board = board.clone()
        test_board[0, test_place_i, test_place_j] = 1
        test_board[1:, test_place_i, test_place_j] = piece
        if checkAlign(test_board):
            return test_place_i, test_place_j
    return None


def heuristicGive(board, pieces):
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
