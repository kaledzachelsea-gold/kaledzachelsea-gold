
import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """Returns the starting state of the board."""
    return [
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY]
    ]


def player(board):
    """Returns the player who has the next turn."""
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    if x_count <= o_count:
        return X
    return O


def actions(board):
    """Returns all possible actions on the board."""
    possible_moves = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_moves.add((i, j))

    return possible_moves


def result(board, action):
    """Returns the board after making the given move."""
    i, j = action

    if action not in actions(board):
        raise ValueError("Invalid action")

    new_board = deepcopy(board)
    new_board[i][j] = player(board)

    return new_board


def winner(board):
    """Returns the winner if there is one."""

    # Check rows
    for row in board:
        if row[0] is not EMPTY and row[0] == row[1] == row[2]:
            return row[0]

    # Check columns
    for j in range(3):
        if board[0][j] is not EMPTY and board[0][j] == board[1][j] == board[2][j]:
            return board[0][j]

    # Check diagonals
    if board[0][0] is not EMPTY and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]

    if board[0][2] is not EMPTY and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]

    return None


def terminal(board):
    """Returns True if the game is over."""
    if winner(board) is not None:
        return True

    return all(cell is not EMPTY for row in board for cell in row)


def utility(board):
    """Returns the utility of the board."""
    win = winner(board)

    if win == X:
        return 1
    elif win == O:
        return -1

    return 0


def minimax(board):
    """Returns the optimal action for the current player."""

    if terminal(board):
        return None

    def max_value(state, alpha, beta):
        if terminal(state):
            return utility(state)

        value = -math.inf

        for action in actions(state):
            value = max(
                value,
                min_value(result(state, action), alpha, beta)
            )

            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value

    def min_value(state, alpha, beta):
        if terminal(state):
            return utility(state)

        value = math.inf

        for action in actions(state):
            value = min(
                value,
                max_value(result(state, action), alpha, beta)
            )

            beta = min(beta, value)

            if alpha >= beta:
                break

        return value

    if player(board) == X:
        best_value = -math.inf
        best_action = None

        for action in actions(board):
            value = min_value(result(board, action), -math.inf, math.inf)

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    else:
        best_value = math.inf
        best_action = None

        for action in actions(board):
            value = max_value(result(board, action), -math.inf, math.inf)

            if value < best_value:
                best_value = value
                best_action = action

        return best_action
