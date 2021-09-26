from Board import PieceColor
from enum import Enum
import os.path

from npBoard import npBoard
import numpy as np
# NOTE: Blue is first place

BOARD_SIZE = 8

def main():
    gameOver = False
    gameboard = npBoard()
    myColor = PieceColor.ORANGE
    theirColor = PieceColor.BLUE
    while(not gameOver):

        # if game is over break
        if(os.path.isfile('end_game')):
            print('GG EZ')  # TODO remove for improved runtime
            gameOver = True
            continue

        # if not my turn break
        if(not os.path.isfile(__file__ + '.go')):
            print('passing')  # TODO remove for improved runtime
            # maybe add move scanning here to save time?
            # or start caculating possable furture moves
            continue

        # read other players move
        file = open("move_file", "r")
        line = ""
        for next_line in file.readlines():
            if next_line.isspace():
                break
            else:
                line = next_line

        # check to see if you are making the first move
        # aka no move before this one
        if line == "":
            print("Let me go first")  # TODO remove for improved runtime
            theirColor = PieceColor.ORANGE
            myColor = PieceColor.BLUE
        else:  # if there is a move that exists from the oponet do it
            # Tokenize move
            tokens = line.split()
            col = tokens[1]
            row = tokens[2]
            # update internal board
            gameboard.set_piece_coords(int(row), col, -1)

        print(gameboard.to_str([]))  # TODO remove for improved runtime
        # Find all legal moves

        # move making logic
        bestMove = search(gameboard)
        gameboard.board = gameboard.set_piece_index(bestMove, 1)

        # send move
        file = open('move_file', 'w')
        file.write(__file__ + npBoard.writeCoords(bestMove))
        file.close()


def miniMax(gameboard: npBoard):
    """
    Implementation of the minimax algorithm with alpha beta pruning
    :param gameboard is the game board
    :return the optimal move
    """
    # 1 is our piece, -1 is opponent piece, 0 is empty spot

    # get legal moves after
    legalMoves = getLegalmoves(gameboard, 1)
    # row: int, _col: str, color: int)

    # set_piece to do each move
    tree = list()
    for i in legalMoves:
        coords = getCoordsFromIndex(i)
        gameboard.set_piece(row=coords[0], col=coords[1], color=1)
        tree.append(tuple(search(gameboard), i))
    # get legal moves again for opponent moves, set_piece for all of those and run heuristic to get board state value
    # return that heuristic value then run minimax aglo on that
    bestMove = tuple(np.NINF)
    for x in tree:
        if bestMove[0] >= x[0]:
            bestMove = x
    # return index of best value
    return getCoordsFromIndex(bestMove[1])


def heuristic(currBoard: npBoard):
    """
    :param currBoard is the current board state
    :return the heuristic score of the board currently from our POV
    """
    spotWeights = np.array([2, 1, 1, 1, 1, 1, 1, 2,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            2, 1, 1, 1, 1, 1, 1, 2, ])

    return np.sum(currBoard * spotWeights)


def search(gameboard: npBoard):
    """
    Implementation of the search algorithm upon tree of moves\
    :param currBoard is the current board state
    :return the legal moves heuristics of a board state
    """
    bestMove = -1
    bestHeuristic = -1
    legalMoves = npBoard.getLegalmoves(-1, gameboard.getBoard())
    for i in legalMoves:
        tempBoard = gameboard.set_piece_index(index=i, color=-1)
        tempHeuristic = heuristic(tempBoard)
        if bestHeuristic > tempHeuristic:
            bestHeuristic = tempHeuristic
            bestMove = i
    return bestMove

main()  # run code
