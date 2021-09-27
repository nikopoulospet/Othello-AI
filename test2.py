from Board import PieceColor
from enum import Enum
import os.path
import time
from npBoard import npBoard
import numpy as np
# NOTE: Blue is first place

BOARD_SIZE = 8
DEPTH_SEARCH = 3
time_limit = 1000


def main():
    gameOver = False
    gameboard = npBoard()
    while(not gameOver):

        # if game is over break
        if(os.path.isfile('end_game')):
            print('GG EZ')  # TODO remove for improved runtime
            gameOver = True
            continue

        # if not my turn break
        if(not os.path.isfile(__file__ + '.go')):
            time.sleep(0.05)
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
            gameboard.switchToFirstPlayer()
        else:  # if there is a move that exists from the oponet do it
            # Tokenize move
            tokens = line.split()
            player = tokens[0]
            if(player == "test2.py"):
                continue
            col = tokens[1]
            row = tokens[2]
            # update internal board
            if col != "P":
                gameboard.board = npBoard.set_piece_coords(
                    int(row), col, -1, gameboard.board)

        # print(gameboard.to_str([]))  # TODO remove for improved runtime
        # Find all legal moves

        print("test 2 is making a move starting at this state, self is red")
        print(npBoard.to_str(gameboard.board, []))
        # move making logic
        bestMove = miniMax(gameboard)
        gameboard.board = npBoard.set_piece_index(bestMove, 1, gameboard.board)

        # send move
        file = open('move_file', 'w')
        print("test 2 is making the following move")
        print("index: " + str(bestMove) + " Cords:" +
              npBoard.writeCoords(bestMove))
        file.write("test2.py" + npBoard.writeCoords(bestMove))
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

    # set_piece to do each move
    # NOTE: best move is in the form (index of move, huristic of move)
    bestMove = (-9999999, -9999999)
    for i in legalMoves:
        # make our move then send it
        tempBoard = npBoard.set_piece_index(i, 1, gameboard.board)
        best, bestHeuristic = alphaBetaSearch(gameboard)
        # best, bestHeuristic = depthLimitedSearch(tempBoard, bestMove[1], 1)
        if(best != -1):  # if the branch wasnt pruned
            lastMove = (i, bestHeuristic)
            if lastMove[1] >= bestMove[1]:
                bestMove = lastMove
        else:  # testing only delete this else
            print("Pruned Branch")
    # get legal moves again for opponent moves, set_piece for all of those and run heuristic to get board state value
    # return that heuristic value then run minimax aglo on that
    # return index of best value
    return bestMove[0]


def heuristic(currBoard: npBoard):
    """
    :param currBoard is the current board state
    :return the heuristic score of the board currently from our POV
    """
    # spotWeights = np.array([2, 1, 1, 1, 1, 1, 1, 2,
    #                         1, 1, 1, 1, 1, 1, 1, 1,
    #                         1, 1, 1, 1, 1, 1, 1, 1,
    #                         1, 1, 1, 1, 1, 1, 1, 1,
    #                         1, 1, 1, 1, 1, 1, 1, 1,
    #                         1, 1, 1, 1, 1, 1, 1, 1,
    #                         1, 1, 1, 1, 1, 1, 1, 1,
    #                         2, 1, 1, 1, 1, 1, 1, 2, ])

    # return np.sum(currBoard * spotWeights)

    # Legal moves worth 10
    # Corners worth 100
    # B2, B7, G2, and G7 worth -25
    ourLegalMoves = len(npBoard.getLegalmoves(1, currBoard))
    theirLegalMoves = len(npBoard.getLegalmoves(-1, currBoard))
    moveWeight = ourLegalMoves - theirLegalMoves

    ourDiscs = npBoard.getPlayerPositions(1, currBoard)
    theirDiscs = npBoard.getPlayerPositions(-1, currBoard)
    discWeight = len(ourDiscs) - len(theirDiscs)

    spotWeights = np.array([100, 1, 1, 1, 1, 1, 1, 100,
                            1, -50, 1, 1, 1, 1, -50, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            1, -50, 1, 1, 1, 1, -50, 1,
                            100, 1, 1, 1, 1, 1, 1, 100, ])
    spotWeight = np.sum(currBoard*spotWeights)
    return discWeight + spotWeight + moveWeight


def search(gameboardArray, pruningValue):
    """
    Implementation of the search algorithm upon tree of moves
    :param currBoard is the current board state
    :return the legal moves heuristics and index of the move or -1 for the index if the branch was prunded
    """
    bestMove = 9999999
    bestHeuristic = 9999999
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)
    # print(npBoard.to_str(gameboardArray, legalMoves))
    for i in legalMoves:
        tempBoard = npBoard.set_piece_index(
            index=i, color=-1, board=gameboardArray)
        tempHeuristic = heuristic(tempBoard)
        # alpha beta pruning
        if tempHeuristic < pruningValue:
            return -1, pruningValue
        if bestHeuristic > tempHeuristic:
            bestHeuristic = tempHeuristic
            bestMove = i
    return bestMove, bestHeuristic


def depthLimitedSearch(gameboardArray, pruningValue, depth):
    """
    Implementation of the search algorithm upon tree of moves
    :param currBoard is the current board state
    :return the legal moves heuristics and index of the move or -1 for the index if the branch was prunded
    """
    bestMove = 9999999
    bestHeuristic = 9999999
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)
    if depth == DEPTH_SEARCH:
        for i in legalMoves:
            tempBoard = npBoard.set_piece_index(
                index=i, color=-1, board=gameboardArray)
            tempHeuristic = heuristic(tempBoard)
            # alpha beta pruning
            if tempHeuristic < pruningValue:
                return -1, pruningValue
            if tempHeuristic < bestHeuristic:
                bestHeuristic = tempHeuristic
                bestMove = i
        return bestMove, bestHeuristic
    else:
        # TODO : rename to bestMove, bestHeuristic?
        val1, val2 = minLayer(gameboardArray, pruningValue, depth)
        return val1, val2


def minLayer(gameboardArray, pruningValue, recussionDepth):
    # oppenents legal moves
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)
    bestMove = 9999999
    bestHeuristic = 9999999
    for theirMove in legalMoves:
        # simulate their move
        theirTempBoard = npBoard.set_piece_index(
            theirMove, -1, gameboardArray)
        # find our responses to their move
        bestMoveHere, bestHeuristicMax = maxLayer(
            theirTempBoard, pruningValue, recussionDepth)
        if bestHeuristicMax < pruningValue:
            return -1, pruningValue
        if bestHeuristicMax < bestHeuristic:
            bestHeuristic = bestHeuristicMax
            bestMove = bestMoveHere
    return bestMove, bestHeuristic


def maxLayer(gameboardArray, pruningValue, recussionDepth):
    # our legal moves
    nextLegalMoves = npBoard.getLegalmoves(1, gameboardArray)
    bestMove = -9999999
    bestHeuristic = -9999999
    for ourMove in nextLegalMoves:
        # simulate our move
        tempBoard = npBoard.set_piece_index(ourMove, 1, gameboardArray)
        bestMoveHere, passedHeuristic = depthLimitedSearch(
            tempBoard, pruningValue, recussionDepth + 1)
        if passedHeuristic > bestHeuristic:
            bestHeuristic = passedHeuristic
            bestMove = ourMove
    return bestMove, bestHeuristic


def findMax(gameboardArray, alpha, beta, currDepth):
    """
    Maximize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta 
    :param currDepth is the current depth of the search
    :return currMin is the current minimum heuristic
    """
    if currDepth == DEPTH_SEARCH:
        return heuristic(gameboardArray)
    currMax = np.NINF
    legalMoves = npBoard.getLegalmoves(
        1, gameboardArray)
    for move in legalMoves:
        currMax = max(currMax, findMin(
            gameboardArray, alpha, beta, currDepth+1))
        if currMax >= beta:
            return currMax
        alpha = max(alpha, currMax)
    return currMax


def findMin(gameboardArray, alpha, beta, currDepth):
    """
    Minimize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta 
    :param currDepth is the current depth of the search
    :return currMax is the current maximum heuristic
    """
    if currDepth == DEPTH_SEARCH:
        return heuristic(gameboardArray)
    currMin = np.inf
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)
    for move in legalMoves:
        currMin = min(currMin, findMax(
            gameboardArray, alpha, beta, currDepth+1))
        if currMin <= alpha:
            return currMin
        beta = min(beta, currMin)
    return currMin


def alphaBetaSearch(gameboardArray):
    """
    Depth Limited Search using alpha beta pruning
    :param gameboardArray is the gameboard
    :return bestMove, bestHeuristic is the index and heuristic of the optimal move
    """
    bestMove = 9999999
    bestHeuristic = 9999999
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)
    for move in legalMoves:
        currBest = findMin(gameboardArray, np.NINF, np.inf, 0)
        if currBest > bestHeuristic:
            bestHeuristic = currBest
            bestMove = move
    return bestMove, bestHeuristic


main()  # run code
