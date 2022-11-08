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


def same(modelA, modelB):
    same = True
    for k, v in modelA.state_dict().items():
        if not (torch.all(v == modelB.state_dict()[k])):
            same = False
    return same


def win_loss(proba, played, gamma=0.95):
    g = gamma
    loss = torch.zeros(1)
    for i in range(len(proba)-1, -1, -1):
        loss += (g * F.mse_loss(proba[i], played[i]))
        g *= gamma
    return loss


def lost_loss(proba, played, gamma=0.95):
    g = gamma
    loss = torch.zeros(1)
    for i in range(len(proba)-1, -1, -1):
        loss += (g * F.mse_loss(proba[i], (1-played[i])/16))
        g *= gamma
    return loss


def tie_loss(proba, played, gamma=0.95):
    g = gamma
    loss = torch.zeros(1)
    for i in range(len(proba)-1, -1, -1):
        loss += 0.8 * (g * F.mse_loss(proba[i], (0.8*(1-played[i]))/8))
        g *= gamma
    return loss


def loss_gameA(placeModel, giveModel, stratPlaceB=placeRandom, stratGiveB=giveRandom):
    board = np.zeros((4, 4, 5), dtype=np.uint8)
    pieces = np.zeros((5, 16), dtype=np.uint8)
    pieces[1, 1::2] = 1
    pieces[2, 2::4] = 1
    pieces[2, 3::4] = 1
    pieces[3, 4:8] = 1
    pieces[3, 12:] = 1
    pieces[4, 8:] = 1

    place_proba = []
    place_played = []
    give_proba = []
    give_played = []

    piece_id = 0
    for i in range(7):
        pieces[0, piece_id] = 1
        piece = pieces[1:, piece_id]
        ppr = placeModel.forward_np(board, pieces, piece)
        p = np.random.choice(16, 1, p=ppr.detach().numpy())[0]
        place_id_1, place_id_2 = p//4, p % 4
        ppl = torch.zeros_like(ppr)
        ppl[p] = 1
        place_proba.append(ppr)
        place_played.append(ppl)
        board[place_id_1, place_id_2, 0] = 1
        board[place_id_1, place_id_2, 1:] = piece
        if checkAlign(board):
            # player A wins
            # print('won', i)
            place_loss = win_loss(place_proba, place_played)
            give_loss = win_loss(give_proba, give_played)
            return place_loss, give_loss
        gpr = giveModel.forward_np(board, pieces)
        g = np.random.choice(16, 1, p=gpr.detach().numpy())[0]
        gpl = torch.zeros_like(ppr)
        gpl[g] = 1
        piece_id = g
        give_proba.append(gpr)
        give_played.append(gpl)
        pieces[0, piece_id] = 1
        piece = pieces[1:, piece_id]
        place_id_1, place_id_2 = stratPlaceB(board, pieces, piece)
        board[place_id_1, place_id_2, 0] = 1
        board[place_id_1, place_id_2, 1:] = piece
        if checkAlign(board):
            # player B wins
            # print('lost', i)
            place_loss = lost_loss(place_proba, place_played)
            give_loss = lost_loss(give_proba, give_played)
            return place_loss, give_loss
        piece_id = stratGiveB(board, pieces)

    pieces[0, piece_id] = 1
    piece = pieces[1:, piece_id]
    ppr = placeModel.forward_np(board, pieces, piece)
    p = np.random.choice(16, 1, p=ppr.detach().numpy())[0]
    place_id_1, place_id_2 = p//4, p % 4
    ppl = torch.zeros_like(ppr)
    ppl[p] = 1
    place_proba.append(ppr)
    place_played.append(ppl)
    board[place_id_1, place_id_2, 0] = 1
    board[place_id_1, place_id_2, 1:] = piece
    if checkAlign(board):
        # player A wins
        # print('won')
        place_loss = win_loss(place_proba, place_played)
        give_loss = win_loss(give_proba, give_played)
        return place_loss, give_loss

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
        # print('lost')
        place_loss = lost_loss(place_proba, place_played)
        give_loss = lost_loss(give_proba, give_played)
        return place_loss, give_loss

    # no winner
    # print('tie')
    place_loss = tie_loss(place_proba, place_played)
    give_loss = tie_loss(give_proba, give_played)
    return place_loss, give_loss


def trainA(placeModel, giveModel, placeOptim, giveOptim, stratPlaceB=placeRandom, stratGiveB=giveRandom, epochs=101, games_in_epoch=100):
    placeModel.train()
    giveModel.train()
    place_losses = []
    give_losses = []

    for epoch in range(epochs):
        if epoch % 10 == 0:
            testA(placeModel, giveModel, N=10**3)

        placeOptim.zero_grad()
        placeOptim.zero_grad()
        place_loss = torch.zeros(1)
        give_loss = torch.zeros(1)
        for i in range(games_in_epoch):
            pl, gl = loss_gameA(placeModel, giveModel, stratPlaceB, stratGiveB)
            place_loss += pl
            give_loss += gl
        place_losses.append(place_loss)
        give_losses.append(give_loss)

        place_loss.backward()
        placeOptim.step()
        give_loss.backward()
        giveOptim.step()
        if epoch % 5 == 0:
            print(f'Train Epoch: {epoch}', end='\t\t')
            print(
                f'Place loss: {place_loss[0]:.2f} ; Give loss: {give_loss[0]:.2f}')
    return place_losses, give_losses


def testA(placeModel, giveModel, stratPlace=placeRandom, stratGive=giveRandom, N=10**3):
    placeModel.eval()
    giveModel.eval()

    with torch.no_grad():
        w, t, l = games(placeModel.play_rand,
                        giveModel.play_rand, stratPlace, stratGive, N)
    print(f"wins: {w} ({100*w/N}%)", end=' ; ')
    print(f"ties: {t} ({100*t/N}%)", end=' ; ')
    print(f"losts: {l} ({100*l/N}%)", end=' ; ')
    print(f"score: {w-l} ({(w-l)/N})")
    return w, t, l
