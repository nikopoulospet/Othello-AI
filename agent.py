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
DEPTH_LIMIT = 1
time_limit = 1000
wentFirst = False
movesVisited = {}


def main():
    gameOver = False
    gameboard = npBoard()
    moveTimer = np.array([1])
    while(not gameOver):
        wentFirst = False
        # if game is over break
        if(os.path.isfile('end_game')):
            gameOver = True
            continue

        # if not my turn break
        if(not os.path.isfile(__file__ + '.go')):
            # time.sleep(0.05)
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
            wentFirst = True
            # global DEPTH_LIMIT
            # DEPTH_LIMIT = 3
            gameboard.switchToFirstPlayer()
        else:  # if there is a move that exists from the oponet do it
            # Tokenize move
            tokens = line.split()
            player = tokens[0]
            if(player == "agent.py"):
                continue
            col = tokens[1]
            row = tokens[2]
            # update internal board
            if col != "P":
                gameboard.board = npBoard.set_piece_coords(
                    int(row), col, -1, gameboard.board)

        # Find all legal moves
        # move making logic
        t2_start = process_time_ns()
        # goodMoves = shallowSearch(gameboard)
        bestMove = miniMax(gameboard)
        t2_stop = process_time_ns()
        t2_diff = (t2_stop - t2_start) / 1000000000
        moveTimer = np.append(moveTimer, [[t2_diff]])
        gameboard.board = npBoard.set_piece_index(bestMove, 1, gameboard.board)

        # send move
        file = open('move_file', 'w')
        file.write("agent.py" + npBoard.writeCoords(bestMove))
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

    # check to see if passing is needed
    if len(legalMoves) == 0:
        return -1

    bestMove = np.inf
    bestHeuristic = np.NINF
    # look at all the possible responses we have to the opponents move
    max_time = int(3)
    start_time = time.time()  # remember when we started
    # while (time.time() - start_time) < max_time:
    for move in legalMoves:
        currBest = findMin(
            gameboard.board, bestHeuristic, bestMove, 0, DEPTH_LIMIT)
        if currBest > bestHeuristic:
            bestHeuristic = currBest
            bestMove = move
    return bestMove


def evaluation(currBoard: npBoard):
    """
    :param currBoard is the current board state
    :return the heuristic score of the board currently from our POV
    """
    # Legal moves worth 10
    # Corners worth 100
    # B2, B7, G2, and G7 worth -25

    if 64 - np.sum(np.abs(currBoard)) <= DEPTH_LIMIT * 2:
        return np.sum(currBoard)

    ourLegalMoves = len(npBoard.getLegalmoves(1, currBoard))
    theirLegalMoves = len(npBoard.getLegalmoves(-1, currBoard))
    moveWeight = ourLegalMoves - theirLegalMoves

    discWeight = np.sum(currBoard)

    spotWeights = np.array([4, -3, 2, 2, 2, 2, -3, 4,
                            -3, -4, -1, -1, -1, -1, -4, -3,
                            2, -1, 1, 0, 0, 1, -1, 2,
                            2, -1, 0, 1, 1, 0, -1, 2,
                            2, -1, 0, 1, 1, 0, -1, 2,
                            2, -1, 1, 0, 0, 1, -1, 2,
                            -3, -4, -1, -1, -1, -1, -4, -3,
                            4, -3, 2, 2, 2, 2, -3, 4])

    spotWeight = np.sum(currBoard*spotWeights)

    return discWeight * -0.25 + spotWeight / 40 + moveWeight / 10


def heuristic(currBoard: npBoard):
    return len(npBoard.getLegalmoves(-1, currBoard))


def findMax(gameboardArray, alpha, beta, currDepth, depthLimit):
    """
    Maximize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta
    :param currDepth is the current depth of the search
    :return currMin is the current minimum heuristic
    """
    if currDepth == depthLimit:
        return evaluation(gameboardArray)
    currMax = np.NINF
    legalMoves = npBoard.getLegalmoves(1, gameboardArray)
    if not legalMoves:
        return evaluation(gameboardArray)
    # orderedMoves = orderMoves(gameboardArray, legalMoves)
    for move in legalMoves:
        # if str(np.append(gameboardArray, move)) in movesVisited:
        #     continue
        # movesVisited[str(np.append(gameboardArray, move))] = 1
        currMax = max(currMax, findMin(
            npBoard.set_piece_index(move, 1, gameboardArray), alpha, beta, currDepth+1, depthLimit))
        if currMax >= beta:
            return currMax
        alpha = max(alpha, currMax)
    return currMax


def findMin(gameboardArray, alpha, beta, currDepth, depthLimit):
    """
    Minimize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta
    :param currDepth is the current depth of the search
    :return currMax is the current maximum heuristic
    """
    # if we already balls deep
    if currDepth == depthLimit:
        return evaluation(gameboardArray)

    currMin = np.inf
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)
    if not legalMoves:
        return evaluation(gameboardArray)
    # explore the opontents counter moves to the one we were thinking of making
    # orderedMoves = orderMoves(gameboardArray, legalMoves)
    for move in legalMoves:
        # if str(np.append(gameboardArray, move)) in movesVisited:
        #     continue
        # movesVisited[str(np.append(gameboardArray, move))] = 1
        currMin = min(currMin, findMax(
            npBoard.set_piece_index(move, 1, gameboardArray), alpha, beta, currDepth+1, depthLimit))
        if currMin <= alpha:  # prune
            return currMin
        beta = min(beta, currMin)
    return currMin


def orderMoves(gameboardArray, moves: list):
    ordered = []
    for move in moves:
        ordered.append((move, evaluation(move)))
    ordered.sort(key=lambda x: x[1], reverse=True)
    return ordered


t1_start = process_time_ns()
main()  # run code
t1_stop = process_time_ns()
