import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from game import *
from tournament import *
from random_strat import *
from give_strat import *
from place_strat import *


class PlaceNet(nn.Module):
    def __init__(self):
        super(PlaceNet, self).__init__()
        self.pieceIn = nn.Linear(4, 256)
        self.boardIn = nn.Linear(80, 256)
        self.piecesIn = nn.Linear(16, 256)
        self.middle = nn.Linear(256, 64)
        self.placeOut = nn.Linear(64, 16)

    def forward(self, b_top, b, pieces, piece):
        x0 = self.pieceIn(piece)
        x1 = self.boardIn(b)
        y = self.piecesIn(pieces)
        z = F.relu(x0)+F.relu(x1)+F.relu(y)
        z = F.relu(self.middle(z))
        place_out = F.softmax(self.placeOut(z), dim=0)
        legal_place_out = F.normalize(
            place_out*(1-b_top.flatten()), p=1, dim=0)
        return legal_place_out

    def forward_np(self, board, pieces, piece):
        b = np.hstack((
            np.sum(board[:, :, 0] *
                   np.moveaxis(board[:, :, 1:], 2, 0), axis=2).flatten(),
            np.sum(board[:, :, 0] *
                   np.moveaxis(board[:, :, 1:], 2, 0), axis=1).flatten(),
            np.sum(np.diagonal(
                board[:, :, 0]*np.moveaxis(board[:, :, 1:], 2, 0), 0, 1, 2), axis=1),
            np.sum(np.diagonal(
                board[:, ::-1, 0]*np.moveaxis(board[:, ::-1, 1:], 2, 0), 0, 1, 2), axis=1),
            np.sum(board[:, :, 0]*np.moveaxis(1 -
                   board[:, :, 1:], 2, 0), axis=2).flatten(),
            np.sum(board[:, :, 0]*np.moveaxis(1 -
                   board[:, :, 1:], 2, 0), axis=1).flatten(),
            np.sum(np.diagonal(
                board[:, :, 0]*np.moveaxis(1-board[:, :, 1:], 2, 0), 0, 1, 2), axis=1),
            np.sum(np.diagonal(
                board[:, ::-1, 0]*np.moveaxis(1-board[:, ::-1, 1:], 2, 0), 0, 1, 2), axis=1)
        )).astype(np.int8)
        return self.forward(torch.Tensor(board[:, :, 0]), torch.Tensor(b), torch.Tensor(pieces[0, :]), torch.Tensor(piece))

    def play_rand(self, board, pieces, piece):
        out = self.forward_np(board, pieces, piece)
        p = np.random.choice(16, 1, p=out.detach().numpy())[0]
        return p//4, p % 4


class GiveNet(nn.Module):
    def __init__(self):
        super(GiveNet, self).__init__()
        self.boardIn = nn.Linear(80, 256)
        self.piecesIn = nn.Linear(16, 256)
        self.middle = nn.Linear(256, 64)
        self.giveOut = nn.Linear(64, 16)

    def forward(self, b, pieces):
        x = self.boardIn(torch.flatten(b))
        y = self.piecesIn(pieces)
        z = F.relu(x)+F.relu(y)
        z = F.relu(self.middle(z))
        give_out = F.softmax(self.giveOut(z), dim=0)
        legal_give_out = F.normalize(give_out*(1-pieces), p=1, dim=0)
        return legal_give_out

    def forward_np(self, board, pieces):
        b = np.hstack((
            np.sum(board[:, :, 0] *
                   np.moveaxis(board[:, :, 1:], 2, 0), axis=2).flatten(),
            np.sum(board[:, :, 0] *
                   np.moveaxis(board[:, :, 1:], 2, 0), axis=1).flatten(),
            np.sum(np.diagonal(
                board[:, :, 0]*np.moveaxis(board[:, :, 1:], 2, 0), 0, 1, 2), axis=1),
            np.sum(np.diagonal(
                board[:, ::-1, 0]*np.moveaxis(board[:, ::-1, 1:], 2, 0), 0, 1, 2), axis=1),
            np.sum(board[:, :, 0]*np.moveaxis(1 -
                   board[:, :, 1:], 2, 0), axis=2).flatten(),
            np.sum(board[:, :, 0]*np.moveaxis(1 -
                   board[:, :, 1:], 2, 0), axis=1).flatten(),
            np.sum(np.diagonal(
                board[:, :, 0]*np.moveaxis(1-board[:, :, 1:], 2, 0), 0, 1, 2), axis=1),
            np.sum(np.diagonal(
                board[:, ::-1, 0]*np.moveaxis(1-board[:, ::-1, 1:], 2, 0), 0, 1, 2), axis=1)
        )).astype(np.int8)
        return self.forward(torch.Tensor(b), torch.Tensor(pieces[0, :]))

    def play_rand(self, board, pieces):
        out = self.forward_np(board, pieces)
        g = np.random.choice(16, 1, p=out.detach().numpy())[0]
        return g
