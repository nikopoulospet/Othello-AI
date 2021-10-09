import os.path
import numpy as np
from multiprocessing import Process, Event, Queue
from time import sleep, time_ns
from enum import Enum
from numpy.core.numeric import Inf
from math import floor

# NOTE: Blue is first place

BOARD_SIZE = 8
DEPTH_LIMIT = 6
TIME_LIMIT = 4
NUM_THREADS = 15
TIME_PERCENT = .98  # becoming unstable around .9
CUT_LOSSES_PERCENT = .99  # becoming unstable around .9
movesVisited = {}

# Event objects used to send signals to threads
stop_event = Event()
start_event = Event()
kill_thread_event = Event()
# thread input and output variabels
threadInGlobal = [Queue()] * NUM_THREADS
threadOutGlobal = [Queue()] * NUM_THREADS


class Direction(Enum):
    """
    Enum of possible directions on an Othello board
    """
    UP = (-1, 0)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (-1, 1)
    DOWN = (1, 0)
    DOWN_LEFT = (1, -1)
    DOWN_RIGHT = (1, 1)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class npBoard:
    """
    faster and more useable othello 
    """

    def __init__(self):
        self.board = np.zeros(64)
        self.board = npBoard.set_piece_coords(5, 'D', 1, self.board)
        self.board = npBoard.set_piece_coords(5, 'E', -1, self.board)
        self.board = npBoard.set_piece_coords(4, 'D', -1, self.board)
        self.board = npBoard.set_piece_coords(4, 'E', 1, self.board)

    def switchToFirstPlayer(self):
        self.board *= -1

    def getBoard(self):
        return self.board

    def getCoordsFromIndex(move: int):
        """
        Takes in the index of a move 0-63 and returns the cordanits 
        :param move: index of the move to be converted
        :return: row and column
        """
        row: int = -1*((move // BOARD_SIZE) - 8)  # 1-8
        col: str = chr(65+(move % BOARD_SIZE))  # A-H
        return row, col

    def _get_row_col_from_coord(_row: int, _col: str):
        """
        gets the row col representation from a coordnate input (1-8,A-H)
        :param move: index of move in question
        :return: row,column in 0-7,0-7 format
        """
        row = -1*(_row - 8)  # because row in this form is 1-8
        col = ord(_col) - ord('A')
        return row, col

    def _get_row_col_from_index(index: int):
        """
        gets the local coordinate representation of an index (0-7, 0-7)
        :param move: index of move in question
        :return: row,column in 0-7,0-7 format
        """
        row: int = (index // BOARD_SIZE)  # 0-7 range
        col: int = index % BOARD_SIZE  # 0-7 range
        return row, col

    def set_piece_index(index: int, color: int, board=np.array([])):
        """
        Wrapper method for set piece so that you dont need to explictly convert indexes to coords
        :param index_move: index of the move to make
        :param color: Piececolor
        :return: False if illegal move, true otherwise
        """
        if(index == -1):
            return board
        row, col = npBoard._get_row_col_from_index(index)
        return npBoard._set_piece(row, col, color, board)

    def set_piece_coords(_row: int, _col: str, color: int, board=np.array([])):
        """
        Set piece at the given coordinates to the given color (1 = us, -1 = them), takes in coords from the (int, str) format
        :param row: Row in the form 1-8
        :param col: Column in the form A-H
        :param color: PieceColor
        :return: False if this was an illegal move for some reason, otherwise True
        """
        if(_col == "P"):
            return board
        row, col = npBoard._get_row_col_from_coord(_row, _col)
        return npBoard._set_piece(row, col, color, board)

    def _set_piece(row: int, col: str, color: int, board):
        """
        Set piece at the given coordinates to the given color (1 = us, -1 = them), generic form of set piece that abstracts the index args
        :param row: Row in the form 0-7
        :param col: Column in the form 0-7
        :param color: PieceColor
        """
        copyBoard = np.copy(board)
        # Check if coordinates are out of bounds
        if npBoard._out_of_bounds(row, col):
            print("board coords are out of bounds")
            print("Trying to go out of bounds: ",
                  npBoard._get_row_col_from_coord(row, col))
            return copyBoard

        # Check if space is occupied
        if copyBoard[row * 8 + col] != 0:
            print("board coords are already occupied")
            print("Trying to occupy: ", npBoard._get_row_col_from_coord(row, col))
            return copyBoard

        # Make change to board
        copyBoard[row * 8 + col] = color

        # Check for envelopment
        envelop = npBoard._get_enveloped_pieces(row, col, color, copyBoard)
        for coords in envelop:
            copyBoard[coords[0] * 8 + coords[1]] = color

        return copyBoard

    def _get_enveloped_pieces(row: int, col: int, color: int, board):
        """
        Get all pieces that would be enveloped if a piece of the given color were placed at the given coordinates
        :param row: Row in the form 0-7
        :param col: Column in the form 0-7
        :param color: Piece color to attempt to envelop with
        :return: List of all pieces that would be enveloped if the piece were placed here
        """

        # List of all pieces that would be enveloped
        enveloped = []

        # Iterate over every direction from the given location
        for (row_off, col_off) in [d.value for d in Direction]:
            row_curr = row + row_off
            col_curr = col + col_off
            # List of pieces that could potentially be enveloped based on information known thus far
            potential_flip = []
            envelop = False
            while not npBoard._out_of_bounds(row_curr, col_curr):

                # Check if piece could be enveloped
                color_curr = board[row_curr*8 + col_curr]
                if color_curr == 0:
                    break
                elif color_curr == color:
                    # If we hit one or more pieces of the opposite color then a piece of the same color, they are
                    # enveloped
                    if len(potential_flip) > 0:
                        envelop = True
                    break
                else:
                    potential_flip.append([row_curr, col_curr])

                # Offset coordinates in direction
                row_curr += row_off
                col_curr += col_off

            # If our list of potentially enveloped pieces is actually enveloped, add it to running list.
            if envelop:
                enveloped.extend(potential_flip)

        return enveloped

    def _out_of_bounds(row: int, col: int) -> bool:
        """
        Check if coordinates are out of bounds, MUST BE (0-7,0-7)
        :param row: Integer row coordinate
        :param col: Integer column coordinate
        :return: True if out of bounds, false if on board
        """
        if (row < 0) or (row > 7) or (col < 0) or (col > 7):
            return True
        else:
            return False

    def getPlayerPositions(nextPiece: int, board):
        """
        Compiles a list of all player positions in local 0-7, 0-7 coordinates
        :param nextPiece: int of player in question
        :return: a list of positions in local 0-7, 0-7 coordinates
        """
        interestSpots = list()
        interestSpots = [i[0]
                         for i, v in np.ndenumerate(board) if v == nextPiece]
        return [npBoard._get_row_col_from_index(i) for i in interestSpots]

    def getLegalmoves(nextPiece: int, board):
        """
        Takes in a game board and the player who is about to move then returns a list of all legal moves
        :param Board: current board
        :param int: the color of the peice that is about to move (-1 for opponent 1 for us)
        :return: list of the index of all legal moves
        """
        # iterate through board, if selected piece, then propogate out
        # use a set because duplicate checking is O(1)
        legalMoves = set()
        interestSpots = npBoard.getPlayerPositions(nextPiece, board)

        for piece in interestSpots:
            for step_row, step_col in [d.value for d in Direction]:
                # choose a search dir and compute a step for each iteration, iterate first and stop if we reach an edge
                row_ptr = piece[0] + step_row
                col_ptr = piece[1] + step_col
                readyForMove = False
                while not npBoard._out_of_bounds(row_ptr, col_ptr):
                    if(board[row_ptr * 8 + col_ptr] == 0):
                        # seen empty space, either break or add move
                        if(readyForMove):
                            legalMoves.add(row_ptr * 8 + col_ptr)
                        break
                    elif(board[row_ptr * 8 + col_ptr] == nextPiece):
                        # seen own piece, break propogation
                        break
                    else:
                        # seen enemy piece, next step could be valid move
                        readyForMove = True
                    row_ptr += step_row
                    col_ptr += step_col

        return list(legalMoves)

    def writeCoords(index: int):
        """
        generates string from index to write to move file
        """
        if(index == -1):
            return " P 1"
        row, col = npBoard.getCoordsFromIndex(index)
        return " " + col + " " + str(row)

    def to_str(board, pot_moves: list()) -> str:
        """
        Convert this board to a pretty-printed string!
        :return: Pretty-printed string displaying board
        """
        out = ""
        for i in range(8):
            out += str(8 - i) + "  "
            for j in range(8):
                color = board[i * 8 + j]
                temp = "0"
                if color == 1:
                    temp = TerminalColor.RED + "R" + TerminalColor.NRM
                elif color == -1:
                    temp = TerminalColor.BLUE + "B" + TerminalColor.NRM
                if (i*8 + j) in pot_moves:
                    if not temp == "0":
                        temp = TerminalColor.YELLOW + "X" + TerminalColor.NRM
                    else:
                        temp = TerminalColor.GREEN + "X" + TerminalColor.NRM
                out += temp
                out += "  "
            out += "\n"
        out += "   A  B  C  D  E  F  G  H"
        return out


def managedMiniMaxThread(queIN: Queue, queOUT: Queue, stop_event: Event, start_event: Event, kill_thread_event: Event):
    """
    Function to manage the many subprocesses that are computing minimax
    :param queIN this processes incoming data
    :param queOUT this processes outgoing data
    :param stop_event the event to stop calculations 
    :param start_event the event to start calculations 
    :param kill_thread_event the event shut down the process
    :return None
    """
    # if its not time to stop the process
    while not kill_thread_event.is_set():
        # waiting to start
        if not start_event.is_set():
            # clean the queue
            if(not queOUT.empty()):
                queOUT.get()
            sleep(0.1)
        else:
            # if this subprosses is given data then start
            if not queIN.empty():
                # get the input data
                inputData = queIN.get()
                move = inputData[0]
                board = inputData[1]
                # make and start miniMax
                gameboardArray = npBoard.set_piece_index(move, 1, board)
                temp = findMin(gameboardArray, np.NINF, np.inf,
                               0, DEPTH_LIMIT, stop_event)
                # output the data
                queOUT.put((move, temp))
            else:
                sleep(0.1)
    return None


def main():
    gameOver = False
    gameboard = npBoard()

    # premake and start subprocesses
    threadsList = [None] * NUM_THREADS
    for i in range(NUM_THREADS):
        threadsList[i] = Process(target=managedMiniMaxThread, args=(
            threadInGlobal[i], threadOutGlobal[i], stop_event, start_event, kill_thread_event))
        threadsList[i].start()

    while(not gameOver):

        # if game is over end subprocesses and quit
        if(os.path.isfile('end_game')):
            print('GG')
            kill_thread_event.set()
            stop_event.set()
            start_event.clear()
            for i in range(NUM_THREADS):
                threadsList[i].terminate()
                threadsList[i].join()
            gameOver = True
            continue

        # if not my turn break
        if(not os.path.isfile('agentOrder.go')):
            continue

        t1_start = time_ns()
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
            if(player == "agentOrder"):
                continue
            col = tokens[1]
            row = tokens[2]
            # update internal board
            if col != "P":
                gameboard.board = npBoard.set_piece_coords(
                    int(row), col, -1, gameboard.board)

        # move making logic
        bestMove = miniMax(gameboard, t1_start)
        bestMoveString = npBoard.writeCoords(bestMove)
        # send move
        file = open('move_file', 'w')
        file.write("agentOrder" + bestMoveString)
        file.close()
        print("Played the move: " + bestMoveString + "\n")
        stop_event.clear()
        start_event.clear()
        # make move on the board
        gameboard.board = npBoard.set_piece_index(bestMove, 1, gameboard.board)


def miniMax(gameboard: npBoard, startTime):
    """
    Implementation of the minimax algorithm with alpha beta pruning
    :param gameboard is the game board
    :return the optimal move
    """
    # get legal moves after
    legalMoves = npBoard.getLegalmoves(1, gameboard.getBoard())
    numLegalMoves = len(legalMoves)

    # check to see if we need to pass or if there is one move
    if numLegalMoves == 0:
        return -1
    if numLegalMoves == 1:
        return legalMoves[0]

    # send the subprocesses the move data
    for moveIndex in range(numLegalMoves):
        # send data to the premade threads
        threadInGlobal[moveIndex].put((legalMoves[moveIndex], gameboard.board))

    # start the subprocesses
    start_event.set()

    # calculate times to do events
    eventTime = int(((TIME_LIMIT * TIME_PERCENT) * 1000000000) + startTime)
    abortTime = int(((TIME_LIMIT * CUT_LOSSES_PERCENT)
                    * 1000000000) + startTime)

    # tell all threads to stop
    while(time_ns() < abortTime):
        if(time_ns() >= eventTime and not stop_event.is_set()):
            stop_event.set()
            print("I COMMAND YALL TO STOP")
    print("AFTER END CALL TIME: " + str((time_ns() - startTime)/1000000000))
    start_event.clear()
    outputTemp = list()

    # get data from subprocesses
    for i in range(numLegalMoves):
        while not threadOutGlobal[i].empty():
            outputTemp.append(threadOutGlobal[i].get())

    # find the best move in the data
    bestMove = -1
    bestHur = -99999
    for data in outputTemp:
        if (data[1] > bestHur) and (not data[1] == Inf) and data[0] in legalMoves:
            bestMove = data[0]
            bestHur = data[1]
    return bestMove


def evaluation(currBoard: npBoard):
    """
    :param currBoard is the current board state
    :return the evaluation score of the board currently from our POV
    """

    if 64 - np.sum(np.abs(currBoard)) <= 14:
        return np.sum(currBoard)

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

    return int((discWeight * 0.25 + spotWeight / 40 + moveWeight / 10) * 100)


def findMax(gameboardArray, alpha, beta, currDepth, depthLimit, stop_event):
    """
    Maximize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta
    :param currDepth is the current depth of the search
    :return currMin is the current minimum evaluation
    :param depthLimit is the maximum depth to search to
    :param stop_event a multiprocessing event to say time is up
    """
    # we have reached the end of the tree, return evaluation value
    if currDepth == depthLimit:
        return evaluation(gameboardArray)

    # if we should be stopped
    if stop_event.is_set():
        return np.NINF

    # worst case
    currMax = np.NINF

    # see legal moves on max layer (us)
    legalMoves = npBoard.getLegalmoves(1, gameboardArray)

    # return if legalMoves is empty
    if not legalMoves:
        return evaluation(gameboardArray)

    # check moves
    # orderedMoves = orderMoves(gameboardArray, legalMoves, -1)
    for move in legalMoves:
        # check if time is up and return if it is
        if stop_event.is_set():
            return currMax

        # do min layers
        currMax = max(currMax, findMin(
            npBoard.set_piece_index(move, 1, gameboardArray), alpha, beta, currDepth+1, depthLimit, stop_event))

        # do pruning
        if currMax >= beta:
            return currMax

        # update alpha
        alpha = max(alpha, currMax)
    # clean return
    return currMax


def findMin(gameboardArray, alpha, beta, currDepth, depthLimit, stop_event):
    """
    Minimize level of alphg-beta pruning
    :param gameboardArray is the gameboard
    :param alpha is the current alpha value
    :param beta is the current beta
    :param currDepth is the current depth of the search
    :param depthLimit is the maximum depth to search to
    :param stop_event a multiprocessing event to say time is up
    :return currMax is the current maximum evaluation
    """

    # we have reached the end of the tree, return evaluation value
    if currDepth == depthLimit:
        return evaluation(gameboardArray)

    # if we should be stopped
    if stop_event.is_set():
        return np.inf

    # worst case
    currMin = np.inf

    # see legal moves on min layer (opponent)
    legalMoves = npBoard.getLegalmoves(-1, gameboardArray)
    # if there is no more moves
    if not legalMoves:
        return evaluation(gameboardArray)

    # explore the opontents counter moves to the one we were thinking of making
    # orderedMoves = orderMoves(gameboardArray, legalMoves, -1)
    for move in legalMoves:
        # check if time is up
        if stop_event.is_set():
            return currMin

        # do max layers
        currMin = min(currMin, findMax(
            npBoard.set_piece_index(move, -1, gameboardArray), alpha, beta, currDepth+1, depthLimit, stop_event))

        # do pruning
        if currMin <= alpha:
            return currMin

        beta = min(beta, currMin)
    # clean return
    return currMin


def orderMoves(gameboardArray, moves: list, color: int):
    """
    Order the moves before pruning
    :param gameboardArray is the gameboard
    :param moves is the moves to be ordered
    """
    ordered = []
    if len(moves) == 1:
        ordered.append(
            (moves[0], evaluation(npBoard.set_piece_index(moves[0], color, gameboardArray))))
        return ordered
    for move in moves:
        # create array of tuple : index, heuristic
        ordered.append(
            (move, evaluation(npBoard.set_piece_index(move, color, gameboardArray))))
    # sort by best heuristic value
    if color == 1:
        # maximized
        ordered.sort(key=lambda move: move[1], reverse=True)
    else:
        # minimized
        ordered.sort(key=lambda move: move[1], reverse=False)
    return ordered[:floor(len(ordered)/2)]


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
    main()  # run code
    print("CLEAN DEATH")
