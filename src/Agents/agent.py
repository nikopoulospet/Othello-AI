import os.path
from npBoard import npBoard
import numpy as np
from time import process_time_ns
import time

# NOTE: Blue is first place

BOARD_SIZE = 8
DEPTH_LIMIT = 2
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

        # print(gameboard.to_str([]))  # TODO remove for improved runtime
        # Find all legal moves

        print("Our agent is making a move starting at this state, self is red")
        print(npBoard.to_str(gameboard.board, []))
        # move making logic
        t2_start = process_time_ns()
        # goodMoves = shallowSearch(gameboard)
        bestMove = miniMax(gameboard)
        t2_stop = process_time_ns()
        t2_diff = (t2_stop - t2_start) / 1000000000
        moveTimer = np.append(moveTimer, [[t2_diff]])
        gameboard.board = npBoard.set_piece_index(bestMove, 1, gameboard.board)

        # send move
        print("Our agent is making the following move")
        print("index: " + str(bestMove) + " Cords:" +
              npBoard.writeCoords(bestMove))
        file = open('move_file', 'w')
        file.write("agent.py" + npBoard.writeCoords(bestMove))
        file.close()


def miniMax(gameboard: npBoard, func):
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
    # print("Time passed: ", time.time() - start_time)
    for move in legalMoves:
        currBest = findMin(
            npBoard.set_piece_index(move, 1, gameboard.board), np.NINF,np.inf, 0, DEPTH_LIMIT, func)
        # print("Current best heuristic: ", currBest)
        if currBest > bestHeuristic:
            bestHeuristic = currBest
            bestMove = move
    return bestMove


def evaluation(currBoard: npBoard, func: str):
    """
    :param currBoard is the current board state
    :return the heuristic score of the board currently from our POV
    """
    # Legal moves worth 10
    # Corners worth 100
    # B2, B7, G2, and G7 worth -25

    if func == "norm":
        if 64 - np.sum(np.abs(currBoard)) <= 14:
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

        return discWeight * 0.25 + spotWeight / 40 + moveWeight / 10
    elif func == "disks":
        if 64 - np.sum(np.abs(currBoard)) <= 14:
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

        return discWeight * 0.15 + spotWeight / 40 + moveWeight / 10
    else:
        print("what")

def findMax(gameboardArray, alpha, beta, currDepth, depthLimit, func):
    """
    Maximize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta
    :param currDepth is the current depth of the search
    :return currMin is the current minimum heuristic
    """
    if currDepth == depthLimit:
        return evaluation(gameboardArray, func)
    currMax = np.NINF
    legalMoves = npBoard.getLegalmoves(1, gameboardArray)
    if not legalMoves:
        return evaluation(gameboardArray, func)
    for move in legalMoves:
        currMax = max(currMax, findMin(
            npBoard.set_piece_index(move, 1, gameboardArray), alpha, beta, currDepth+1, depthLimit, func))
        if currMax >= beta:
            return currMax
        alpha = max(alpha, currMax)
    return currMax


def findMin(gameboardArray, alpha, beta, currDepth, depthLimit, func):
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
        return evaluation(gameboardArray,func)

    currMin = np.inf
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)
    if not legalMoves:
        return evaluation(gameboardArray, func)
    # explore the opontents counter moves to the one we were thinking of making
    # , heur in orderMoves(gameboardArray, legalMoves)
    for move in legalMoves:
        currMin = min(currMin, findMax(
            npBoard.set_piece_index(move, -1, gameboardArray), alpha, beta, currDepth+1, depthLimit, func))
        if currMin <= alpha:  # prune
            return currMin
        beta = min(beta, currMin)
    return currMin


def orderMoves(gameboardArray, moves: list, func):
    ordered = []
    for move in moves:
        ordered.append((move, evaluation(gameboardArray, func)))
    ordered.sort(key=lambda move: move[1], reverse=True)
    return ordered

"""
minimax agent wrapper class to use in the gym enviroment. 
Must impliment action = get_action(board) to make steps in gym
"""
class miniMax_agent():
    def __init__(self, search_depth=1, func='norm'):
        self.gameboard = npBoard()
        self.search_depth = search_depth #TODO enforce search_depth in minimax code
        self.func = func
    def get_action(self, observation: np.array([])):
        '''
        action step of the minimax agent, generate a best move
        '''
        # move making logic
        self.gameboard.board = observation
        bestMove = miniMax(self.gameboard, self.func)
        return bestMove

if __name__ == "__main__":
    t1_start = process_time_ns()
    main()  # run code
    t1_stop = process_time_ns()
    print("Elapsed time:", t1_stop, t1_start)
    print("Elapsed time during the whole program in nanoseconds:", t1_stop - t1_start)
