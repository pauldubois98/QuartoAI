import torch


def randomPlace(board):
    available_places = torch.where(board[0, :, :] == 0)
    i = torch.randint(available_places[0].shape[0], (1,))
    return available_places[0][i].item(), available_places[1][i].item()


def randomGive(pieces):
    available_pieces = torch.where(pieces[0, :] == 1)
    i = torch.randint(available_pieces[0].shape[0], (1,))
    return available_pieces[0][i].item()


class RandomPlayer:
    def give(self, board, pieces):
        return randomGive(pieces)

    def place(self, board, pieces, piece):
        return randomPlace(board)
