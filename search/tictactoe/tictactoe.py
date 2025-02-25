"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None

winLines = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],

    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],

    [(0, 0), (1, 1), (2, 2)],
    [(2, 0), (1, 1), (0, 2)]
]


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    numx, numo = (0, 0)
    for x in range(3):
        for y in range(3):
            if (board[x][y] == X):
                numx += 1
            if (board[x][y] == O):
                numo += 1

    if (numo < numx):
        return O 
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for x in range(3):
        for y in range(3):
            if (board[x][y] == EMPTY):
                actions.add((x, y))
    
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if (action[0] < 0 or action[0] > 2):
        raise Exception("Illegal move, out of bounds")
    if (action[1] < 0 or action[1] > 2):
        raise Exception("Illegal move, out of bounds")
    if (board[action[0]][action[1]] != EMPTY):
        raise Exception("Illegal move, occupied cell")

    newBoard = copy.deepcopy(board)
    newBoard[action[0]][action[1]] = player(board)
    return newBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for line in winLines:
        xocc, oocc = (0, 0)
        for cell in line:
            if (board[cell[0]][cell[1]] == X):
                xocc += 1
            if (board[cell[0]][cell[1]] == O):
                oocc += 1
        if (xocc == 3):
            return X
        if (oocc == 3):
            return O

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if (winner(board)):
        return True
    for x in range(3):
        for y in range(3):
            if (board[x][y] == EMPTY):
                return False
    
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    result = winner(board)
    if (result == X):
        return 1
    if (result == O):
        return -1
    return 0


def notPlayer(player):
    if (player == X):
        return O
    return X


def walk(board, plr):
    if terminal(board):
        if (winner(board) == plr):
            return 1
        if (winner(board) == None):
            return 0
        return -1
    
    if (plr == player(board)):
        bestScore = float("-inf")
    else:
        bestScore = float("inf")
    
    for action in actions(board):
        newBoard = result(board, action)
        score = walk(newBoard, plr)
        if (plr == player(board)):
            bestScore = max(score, bestScore)
        else:
            bestScore = min(score, bestScore)

    return bestScore


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if (winner(board)):
        return None
    moves = actions(board)
    if (len(moves) == 9):
        return (1, 1)
    
    bestScore = float('-inf')
    bestMove = None
    for move in moves:
        newRes = result(board, move)
        score = walk(newRes, player(board))
        if (score > bestScore):
            bestScore = score
            bestMove = move
    
    return bestMove
