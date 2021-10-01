import math
import os
import time
import sys
import copy

from Board import Board, PieceColor, transform_coords, interpret_coords

# BLUE = MAX
# ORANGE = MIN
from Game import Game

team_name = "ai.py"

GO_PATH_NAME = team_name + ".go"
END_PATH_NAME = "end_game"
MOVE_PATH_NAME = "move_file"

color = PieceColor.ORANGE

board = Board()


def run():
    while True:
        while not (os.path.exists(GO_PATH_NAME) or os.path.exists(END_PATH_NAME)):
            print("Waiting for turn " + GO_PATH_NAME)
            time.sleep(.05)

        if os.path.exists(END_PATH_NAME):
            print("Game ended")
            exit(1)

        if os.path.exists(GO_PATH_NAME):
            print("It is our turn, making move")
            update_board()
            best_move = get_best_move()
            apply_move(best_move)

            time.sleep(1)


def update_board():
    global color  # wtf
    '''
    Read in the data from move_file and update the board object
    Changes the board state

    IF MOVE_FILE IS EMPTY, CHANGE COLOR TO BLUE ("B")
    :return:
    '''
    with open(MOVE_PATH_NAME, "r") as move_file:

        lines = move_file.readlines()
        print("Contents of move_file: " + str(lines))
        if len(lines) == 0:
            # First move, set to blue
            color = PieceColor.BLUE
        else:
            line = lines[0].strip("\n").split(" ")
            board.set_piece(int(line[2]), line[1], PieceColor.BLUE if color ==
                            PieceColor.ORANGE else PieceColor.ORANGE)
        print(board)


def get_best_move():
    '''
    Computes the best move for the given board, using the AI's color

    Must have a timeout of like 9.5 seconds
    :param board:
    :return: a string for example E 3
    4, 2
    '''

    minimax_result = minimax(board, 3, -100000, 100000, color)
    return minimax_result[1]

    # Pick the first move it sees

    # for row in range(8):
    #    for col in range(8):
    #        piece = board._get_piece(row, col)
    #        if (piece == PieceColor.NONE) and len(board._get_enveloped_pieces(row, col, color)) > 0:
    #            # This is a valid move
    #            print("First seen move is " + str(transform_coords(row, col)))
    #            return transform_coords(row, col)


def heuristic(board, move):
    '''
    Returns the distance of move from the center of the board
    :param board:
    :param move: number then letter "6 E"
    :return:
    '''

    theMove = interpret_coords(move[0], move[1])
    x = theMove[0] - 4
    y = theMove[1] - 4

    return math.sqrt(x ** 2 + y ** 2)


def evaluate_board(board):
    '''
    Return blue disks - orange disks
    :param board:
    :return:
    '''
    orange_disks = 0
    blue_disks = 0
    for row in range(8):
        for col in range(8):
            if board._get_piece(row, col) == PieceColor.ORANGE:
                orange_disks += 1
            elif board._get_piece(row, col) == PieceColor.BLUE:
                blue_disks += 1
    return blue_disks - orange_disks


def expand_board(board, player):
    '''
    Return a list of all possible next board states and the move associated with it
    :param board:
    :return:
    [(board_state, (3, "E")), ....]
    '''
    board_copy = copy.deepcopy(board)
    new_boards = []
    for row in range(8):
        for col in range(8):
            piece = board._get_piece(row, col)
            if(piece == PieceColor.NONE) and len(board._get_enveloped_pieces(row, col, player)) > 0:
                print("Expanding new move: " + str(transform_coords(row, col)))

                new_board = copy.deepcopy(board_copy)
                # print(new_board)
                new_board._set_piece(row, col, player)
                # print(new_board)
                # print("-------")
                new_boards.append((new_board, transform_coords(row, col)))

    return new_boards


def minimax(board_node: Board, depth, alpha, beta, player_color):
    if depth == 0 or board_node.is_full():
        return evaluate_board(board_node), None

    if player_color == PieceColor.BLUE:
        max_eval = -100000
        best_move = ()

        next_boards = expand_board(board_node, player_color)
        next_boards.sort(key=lambda t: heuristic(t[0], t[1]))

        for t in next_boards:
            child_board = t[0]
            move = t[1]

            eval = minimax(child_board, depth-1, alpha,
                           beta, PieceColor.ORANGE)[0]

            #max_eval = max(eval, max_eval)
            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        return max_eval, best_move

    elif player_color == PieceColor.ORANGE:
        min_eval = 100000
        best_move = ()

        next_boards = expand_board(board_node, player_color)
        next_boards.sort(key=lambda t: heuristic(t[0], t[1]))

        for t in next_boards:
            child_board = t[0]
            move = t[1]

            eval = minimax(child_board, depth-1, alpha,
                           beta, PieceColor.BLUE)[0]

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)

            if beta <= alpha:
                break

        return min_eval, best_move


def apply_move(move):
    '''
    First update our board representation

    Then write the move to move_file
    :param move:
    :return:
    '''
    print("Applying move " + str(move))
    board.set_piece(move[0], move[1], color)
    with open(MOVE_PATH_NAME, "w") as move_file:
        move_file.write(team_name + " " + str(move[1]) + " " + str(move[0]))

    # os.remove(GO_PATH_NAME)

    # time.sleep(0.2)


if __name__ == "__main__":
    run()
