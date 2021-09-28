from Board import PieceColor
from enum import Enum
import os.path
import time
from npBoard import npBoard
import numpy as np
from time import process_time_ns

# NOTE: Blue is first place

BOARD_SIZE = 8
DEPTH_LIMIT = 7
time_limit = 1000
wentFirst = False


def main():
    gameOver = False
    gameboard = npBoard()
    moveTimer = np.array([1])
    while(not gameOver):
        wentFirst = False
        # if game is over break
        if(os.path.isfile('end_game')):
            print('GG EZ')  # TODO remove for improved runtime
            if wentFirst:
                print("I WENT FIRST")
            print("Average Move Time: " + str(np.average(moveTimer[1:])))
            print("Longest Move Time: " + str(np.max(moveTimer[1:])))
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
            print("Let me go first")  # TODO remove for improved runtime
            wentFirst = True
            global DEPTH_LIMIT
            DEPTH_LIMIT = 6
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

        # print(gameboard.to_str([]))  # TODO remove for improved runtime
        # Find all legal moves

        print("Our agent is making a move starting at this state, self is red")
        print(npBoard.to_str(gameboard.board, []))
        # move making logic
        t2_start = process_time_ns()
        bestMove = miniMax(gameboard)
        t2_stop = process_time_ns()
        t2_diff = (t2_stop - t2_start) / 1000000000
        moveTimer=np.append(moveTimer, [[t2_diff]])
        gameboard.board = npBoard.set_piece_index(bestMove, 1, gameboard.board)

        # send move
        print("Our agent is making the following move")
        print("index: " + str(bestMove) + " Cords:" +
              npBoard.writeCoords(bestMove))
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
    
    #MASTER MAX LAYER
    bestMove = -1
    bestHeuristic = np.NINF
    legalMoves = npBoard.getLegalmoves(1, gameboard.board)
    #look at all the possable resposise we have to the opnets move
    for move in legalMoves:
        currBest = findMin(gameboard.board, bestHeuristic, bestMove, 0)
        #if the move just simulated was better make it the best move
        if currBest > bestHeuristic:
            bestHeuristic = currBest
            bestMove = move
    print("Huristic Value: " + str(bestHeuristic))
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


def findMax(gameboardArray, alpha, beta, currDepth):
    """
    Maximize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta 
    :param currDepth is the current depth of the search
    :return currMin is the current minimum heuristic
    """
    if currDepth == DEPTH_LIMIT:
        return evaluation(gameboardArray)
    currMax = np.NINF
    legalMoves = npBoard.getLegalmoves(1, gameboardArray)
    if not legalMoves:
        return evaluation(gameboardArray)
    for move in legalMoves:
        currMax = max(currMax, findMin(gameboardArray, alpha, beta, currDepth+1))
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
    #if we already balls deep
    if currDepth == DEPTH_LIMIT:
        return evaluation(gameboardArray)
    
    currMin = np.inf
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)
    if not legalMoves:
        return  evaluation(gameboardArray)
    #explore the opontents counter moves to the one we were thinking of making
    for move in legalMoves:
        currMin = min(currMin, findMax(gameboardArray, alpha, beta, currDepth+1))
        if currMin <= alpha: #prune
            return currMin
        beta = min(beta, currMin)
    return currMin


def alphaBetaSearch(gameboardArray):
    """
    Depth Limited Search using alpha beta pruning
    :param gameboardArray is the gameboard
    :return bestMove, bestHeuristic is the index and heuristic of the optimal move
    """


t1_start = process_time_ns()
main()  # run code
t1_stop = process_time_ns()
print("Elapsed time:", t1_stop, t1_start)

print("Elapsed time during the whole program in nanoseconds:", t1_stop - t1_start)
