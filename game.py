import torch
import numpy as np
import matplotlib.pyplot as plt


def plotPieces(pieces):
    """plot the pieces using hexadecimal numbers (textual representation)"""
    for i in range(pieces.shape[1]):
        if (pieces[0, i] == 0):
            print(f'*{i:x}*', end=' ')
        else:
            print(f' {i:x} ', end=' ')
    print()


def plotBoard(board):
    """plot the board using hexadecimal numbers (textual representation)"""
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
    """initialize the board and the pieces"""
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
    """check if there are 3 pieces of the same shape aligned"""
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


def full_game(playerA, playerB, display=False):
    """play a full game between two players"""
    board, pieces = init_game()
    piece_idx = 0  # exploiting symmetry
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
        if display:
            print('   turn', turn)
            plotBoard(board)
            plotPieces(pieces)


def multiple_games(playerA, playerB, n_games):
    """play multiple games between two players"""
    A_wins = 0
    B_wins = 0
    draws = 0
    for i in range(n_games):
        result = full_game(playerA, playerB, False)
        if result == "A":
            A_wins += 1
        elif result == "B":
            B_wins += 1
        else:
            draws += 1
    return A_wins, B_wins, draws


def tournament(players, n_games):
    """play a tournament between all players"""
    n_players = len(players)
    wins = np.zeros((n_players, n_players), dtype=int)
    losts = np.zeros((n_players, n_players), dtype=int)
    draws = np.zeros((n_players, n_players), dtype=int)
    for i in range(n_players):
        for j in range(n_players):
            wins[i, j], losts[i, j], draws[i, j] \
                = multiple_games(players[i], players[j], n_games)
    tournament_results = {
        "wins": wins,
        "losts": losts,
        "draws": draws,
        "players": players,
        "n_games": n_games,
    }
    return tournament_results


def plot_tournament(tournament_results, output_file=None):
    """plot the results of a tournament"""
    wins = tournament_results["wins"]
    losts = tournament_results["losts"]
    draws = tournament_results["draws"]
    players = tournament_results["players"]
    n_games = tournament_results["n_games"]
    n_players = len(players)
    plt.clf()
    plt.imshow(wins-losts, cmap='bwr_r', vmin=-n_games, vmax=n_games)
    for i in range(n_players):
        for j in range(n_players):
            plt.text(
                j, i, f"{wins[i,j]}; {draws[i,j]} ; {losts[i,j]}", ha='center', va='center')
    plt.title("Tournament results")
    plt.yticks(np.arange(n_players), [player.__str__().replace(
        "Player", "") for player in players])
    plt.xticks(np.arange(n_players), [player.__str__().replace(
        "Player", "") for player in players])
    plt.xticks(rotation=45)
    plt.colorbar()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
