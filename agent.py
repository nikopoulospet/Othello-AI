from Board import PieceColor
from enum import Enum
import os.path
import time
from npBoard import npBoard
import numpy as np
from time import process_time_ns
from itertools import chain
import time

# NOTE: Blue is first place

BOARD_SIZE = 8
DEPTH_LIMIT = 2
time_limit = 1000
movesVisited = {}


def main():
    gameOver = False
    gameboard = npBoard()
    moveTimer = np.array([1])
    while(not gameOver):
        # if game is over break
        if(os.path.isfile('end_game')):
            gameOver = True
            continue

        # if not my turn break
        if(not os.path.isfile('agent.go')):
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
        if line == "":
            # no move from the opponent, we go first
            gameboard.switchToFirstPlayer()
        else:  # if there is a move that exists from the oponet do it
            # Tokenize move
            tokens = line.split()
            player = tokens[0]
            if(player == "agent"):
                continue
            col = tokens[1]
            row = tokens[2]
            # update internal board
            if col != "P":
                gameboard.board = npBoard.set_piece_coords(
                    int(row), col, -1, gameboard.board)

        # move making logic
        bestMove = miniMax(gameboard)

        # make move on the board
        gameboard.board = npBoard.set_piece_index(bestMove, 1, gameboard.board)

        # send move
        file = open('move_file', 'w')
        file.write("agent" + npBoard.writeCoords(bestMove))
        file.close()


def miniMax(gameboard: npBoard):
    """
    Implementation of the minimax algorithm with alpha beta pruning
    :param gameboard is the game board
    :return the optimal move
    """
    # 1 is our piece, -1 is opponent piece, 0 is empty spot

    # get legal moves after
    legalMoves = npBoard.getLegalmoves(1, gameboard.getBoard())

    # check to see if we need to pass
    if len(legalMoves) == 0:
        return -1

    # bestMove is the index of the bestMove
    bestMove = np.inf

    # bestHeuristic is the index of the bestMove
    bestHeuristic = np.NINF

    # start tree with our next possible moves
    for move in legalMoves:
        # start pruning
        currBest = findMin(
            npBoard.set_piece_index(move, 1, gameboard.board), bestHeuristic, bestMove, 0, DEPTH_LIMIT)
        # print("Current best heuristic: ", currBest)
        if currBest > bestHeuristic:
            bestHeuristic = currBest
            bestMove = move
    # return bestMove index
    return bestMove


def evaluation(currBoard: npBoard):
    """
    :param currBoard is the current board state
    :return the evaluation score of the board currently from our POV
    """

    # if 64 - np.sum(np.abs(currBoard)) <= 14:
    #     return np.sum(currBoard)

    # weight between our legal moves and theirs, more legal moves is better
    ourLegalMoves = len(npBoard.getLegalmoves(1, currBoard))
    theirLegalMoves = len(npBoard.getLegalmoves(-1, currBoard))
    moveWeight = ourLegalMoves - theirLegalMoves

    # sum board to see who currently has more discs
    discWeight = np.sum(currBoard)

    # based on othello strategy, weight certain spots more than others. Example: corners are good
    spotWeights = np.array([4, -3, 2, 2, 2, 2, -3, 4,
                            -3, -4, -1, -1, -1, -1, -4, -3,
                            2, -1, 1, 0, 0, 1, -1, 2,
                            2, -1, 0, 1, 1, 0, -1, 2,
                            2, -1, 0, 1, 1, 0, -1, 2,
                            2, -1, 1, 0, 0, 1, -1, 2,
                            -3, -4, -1, -1, -1, -1, -4, -3,
                            4, -3, 2, 2, 2, 2, -3, 4])

    spotWeight = np.sum(currBoard*spotWeights)

    return discWeight * 0.25 + spotWeight / 40 + moveWeight / 10


def findMax(gameboardArray, alpha, beta, currDepth, depthLimit):
    """
    Maximize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta
    :param currDepth is the current depth of the search
    :return currMin is the current minimum evaluation
    """
    # we have reached the end of the tree, return evaluation value
    if currDepth == depthLimit:
        return evaluation(gameboardArray)

    # worst case
    currMax = np.NINF

    # see legal moves on max layer (us)
    legalMoves = npBoard.getLegalmoves(1, gameboardArray)

    # return if legalMoves is empty
    if not legalMoves:
        return evaluation(gameboardArray)
    for move, heur in orderMoves(gameboardArray, legalMoves):
        currMax = max(currMax, findMin(
            npBoard.set_piece_index(move, 1, gameboardArray), alpha, beta, currDepth+1, depthLimit))
        if currMax >= beta:  # prune
            return currMax
        alpha = max(alpha, currMax)  # update alpha
    return currMax


def findMin(gameboardArray, alpha, beta, currDepth, depthLimit):
    """
    Minimize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta
    :param currDepth is the current depth of the search
    :return currMax is the current maximum evaluation
    """
    # we have reached the end of the tree, return evaluation value
    if currDepth == depthLimit:
        return evaluation(gameboardArray)

    # worst case
    currMin = np.inf

    # see legal moves on min layer (opponent)
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)

    if not legalMoves:
        return evaluation(gameboardArray)
    # explore the opontents counter moves to the one we were thinking of making
    for move, heur in orderMoves(gameboardArray, legalMoves):
        currMin = min(currMin, findMax(
            npBoard.set_piece_index(move, -1, gameboardArray), alpha, beta, currDepth+1, depthLimit))
        if currMin <= alpha:  # prune
            return currMin
        beta = min(beta, currMin)  # update beta
    return currMin


def orderMoves(gameboardArray, moves: list):
    """
    Order the moves before pruning
    :param gameboardArray is the gameboard
    :param moves is the moves to be ordered
    """
    ordered = []
    for move in moves:
        ordered.append((move, evaluation(gameboardArray)))
    ordered.sort(key=lambda move: move[1], reverse=True)
    return ordered


"""
minimax agent wrapper class to use in the gym enviroment. 
Must impliment action = get_action(board) to make steps in gym
"""


class miniMax_agent():
    def __init__(self, search_depth=1):
        self.gameboard = npBoard()
        self.search_depth = search_depth  # TODO enforce search_depth in minimax code

    def get_action(self, observation: np.array([])):
        '''
        action step of the minimax agent, generate a best move
        '''
        # move making logic
        self.gameboard.board = observation
        bestMove = miniMax(self.gameboard)
        return bestMove


if __name__ == "__main__":
    t1_start = process_time_ns()
    main()  # run code
    t1_stop = process_time_ns()
    print("Elapsed time:", t1_stop, t1_start)
    print("Elapsed time during the whole program in nanoseconds:", t1_stop - t1_start)
