from game import *
from random_player import *
from heuristic_player import *
import pickle
import random
import torch
import numpy as np
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

n_games = 1000

# aggressivity
players = [RandomPlayer(), PacificPlayer(),
           AggressivePlayer(), HeuristicPlayer()]
tournament_results = tournament(players, n_games)
with open(f'aggressivity-{n_games}.pkl', 'wb') as f:
    pickle.dump(tournament_results, f)

# subheuristics
players = [RandomPlayer(), HeuristicGivePlayer(),
           HeuristicPlacePlayer(), HeuristicPlayer()]
tournament_results = tournament(players, n_games)
with open(f'subheuristics-{n_games}.pkl', 'wb') as f:
    pickle.dump(tournament_results, f)

# heuristics
players = [RandomPlayer(), HeuristicPlayer(),
           AggressiveHeuristicPlayer(), PacificHeuristicPlayer()]
tournament_results = tournament(players, n_games)
with open(f'heuristics-{n_games}.pkl', 'wb') as f:
    pickle.dump(tournament_results, f)
