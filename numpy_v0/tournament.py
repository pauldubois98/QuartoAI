import numpy as np
import pandas as pd
from game import *
from random_strat import *
from give_strat import *
from place_strat import *
from multiprocessing import Pool


def games(stratPlaceA, stratGiveA, stratPlaceB, stratGiveB, N=10**3):
    w = 0
    t = 0
    l = 0
    for i in range(N):
        r = game(stratPlaceA, stratGiveA, stratPlaceB, stratGiveB)
        if r == 1:
            w += 1
        elif r == -1:
            l += 1
        else:  # r==0
            t += 1
    return w, t, l


def tournament(strats_list, N=10**3):
    # scores = np.zeros((len(strats_list), len(strats_list)), dtype=np.int32)
    wins = np.zeros((len(strats_list), len(strats_list)), dtype=np.int32)
    ties = np.zeros((len(strats_list), len(strats_list)), dtype=np.int32)
    lost = np.zeros((len(strats_list), len(strats_list)), dtype=np.int32)
    for i, strat1 in enumerate(strats_list):
        for j, strat2 in enumerate(strats_list):
            wins[i, j], ties[i, j], lost[i, j] = games(
                strat1[0], strat1[1], strat2[0], strat2[1], N)
            print('.', end='')
        print()
    return wins, ties, lost


if __name__ == '__main__':
    N = 10**4

    STRATS = [(placeRandom, giveRandom),
              (placeRandom, giveToNotLose),
              (placeToWin, giveRandom),
              (placeToWin, giveToNotLose)]
    STRATS_NAMES = [strat[0].__name__+' ' +
                    strat[1].__name__ for strat in STRATS]
    # make tournament
    wins, ties, lost = tournament(STRATS, N)
    scores = (wins-lost)/N
    # print results
    for data in ["wins", "ties", "lost", "scores", "N"]:
        df = pd.DataFrame(locals()[data], STRATS_NAMES, STRATS_NAMES)
        df.to_csv(f'{data}-0.csv')

    #######################################################################

    # STRATS = [(placeRandom, giveToNotLose),
    #           (placeToWin, giveToNotLose),
    #           (placeToWin_minTension, giveToNotLose),
    #           (placeToWin_maxTension, giveToNotLose)]
    # STRATS_NAMES = [strat[0].__name__+' ' +
    #                 strat[1].__name__ for strat in STRATS]
    # # make tournament
    # wins, ties, lost = tournament(STRATS, N)
    # scores = (wins-lost)/N
    # # print results
    # for data in ["wins", "ties", "lost", "scores", "N"]:
    #     df = pd.DataFrame(locals()[data], STRATS_NAMES, STRATS_NAMES)
    #     df.to_csv(f'{data}-1.csv')

    #######################################################################

    # STRATS = [(placeRandom, giveRandom),
    #           (minTension, giveRandom),
    #           (maxTension, giveRandom),
    #           (placeToWin, giveRandom)]
    # STRATS_NAMES = [strat[0].__name__+' ' +
    #                 strat[1].__name__ for strat in STRATS]
    # # make tournament
    # wins, ties, lost = tournament(STRATS, N)
    # scores = (wins-lost)/N
    # # print results
    # for data in ["wins", "ties", "lost", "scores", "N"]:
    #     df = pd.DataFrame(locals()[data], STRATS_NAMES, STRATS_NAMES)
    #     df.to_csv(f'{data}-2.csv')
