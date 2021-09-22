from Board import PieceColor
from enum import Enum
import os.path

import numpy as np
# NOTE: Blue is first place

BOARD_SIZE = 8

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
    faster and more uable othello 
    """

    board = np.zeros(64)

    def __init__(self):
        self.set_piece(5, 'D', -1)
        self.set_piece(5, 'E', 1)
        self.set_piece(4, 'D', 1)
        self.set_piece(4, 'E', -1)
        pass

    def switchToFirstPlayer(self):
        self.board * -1

    def getBoard(self):
        return self.board 
    def set_piece(self, row:int, _col:str, color:int):
        """
        Set piece at the given coordinates to the given color (1 = us, -1 = them)
        :param row: Row in the form 0-7
        :param col: Column in the form 0-7
        :param color: PieceColor
        :return: False if this was an illegal move for some reason, otherwise True
        """
        #convert from letter to value
        col = ord(_col) - ord('A')

        # Make change to board
        self.board[row * 8 + col] = color

        # Check for envelopment
        envelop = self._get_enveloped_pieces(row, col, color)
        for coords in envelop:
            self.board[coords[0] * 8 + coords[1]] = color

        # Make sure we enveloped at least one piece, otherwise this was an invalid move
        if len(envelop) == 0:
            return False
        else:
            return True

    def _get_enveloped_pieces(self, row: int, col: int, color: int):
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
            row_curr = row
            col_curr = col

            # List of pieces that could potentially be enveloped based on information known thus far
            potential_flip = []
            envelop = False
            while not out_of_bounds(row_curr, col_curr):
                # Offset coordinates in direction
                row_curr += row_off
                col_curr += col_off

                # Check if piece could be enveloped
                color_curr = self._get_piece(row_curr, col_curr)
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

            # If our list of potentially enveloped pieces is actually enveloped, add it to running list.
            if envelop:
                enveloped.extend(potential_flip)

        return enveloped

    def _get_piece(self, row: int, col: int) -> int:
        """
        Get piece at given coordinates on Othello board
        :param row: Row in the form 0-7
        :param col: Column in the form 0-7
        :return: PieceColor at (row, col) on board
        """
        return self.board[row * 8 + col]

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
            gameboard.set_piece(int(row), col, -1)

        print(gameboard)  # TODO remove for improved runtime
        # Find all legal moves

        # move making logic
        bestMove = search(gameboard, myColor)

        # convert index to move
        row, col = getCoordsFromIndex(bestMove)

        # update model
        gameboard.board.set_piece(int(row), col, myColor)

        # send move
        file = open('move_file', 'w')
        file.write(__file__ + " " + col + " " + row)
        file.close()


def getLegalmoves(gameBoard: npBoard, nextPiece: int):
    """
    Takes in a game board and the player who is about to move then returns a list of all legal moves
    :param Board: current board
    :param int: the color of the peice that is about to move (-1 for opponent 1 for us)
    :return: list of the index of all legal moves
    """
    # iterate through board, if selected piece, then propogate out
    # use a set because duplicate checking is O(1)
    legalMoves = set()
    interestSpots = list()
    interestSpots = [i for i,v in gameBoard.getBoard() if v == nextPiece]
    print(interestSpots)
    for piece in interestSpots:
        for direction in Direction:
            #choose a search dir and compute a step for each iteration, iterate first and stop if we reach an edge
            step = direction.value[0] + (direction.value[1]*BOARD_SIZE)
            searchIndex = piece
            readyForMove = False
            while 1:
                searchIndex += step
                if(gameBoard.board[searchIndex] == PieceColor.NONE):
                    # seen empty space, either break or add move
                    if(readyForMove):
                        legalMoves.add(searchIndex)
                    break
                elif(gameBoard.board[searchIndex] == nextPiece):
                    # seen own piece, break propogation
                    break
                else:
                    # seen enemy piece, next step could be valid move
                    readyForMove = True
                if(onBoardEdge(searchIndex)):
                    #if a piece is on the edge we know there is no more spaces to search
                    break; 
    return list(legalMoves)



def onBoardEdge(searchIndex: int):
    """
    checks if a move is on the edge of the Board
    :params searchIndex: index to be checked
    :return: boolean 
    """
    top = searchIndex >= (BOARD_SIZE*BOARD_SIZE) - BOARD_SIZE
    bot = searchIndex < BOARD_SIZE
    lft = searchIndex % BOARD_SIZE == 0
    rgt = searchIndex+1 % BOARD_SIZE == 0
    return top or bot or lft or rgt

def getCoordsFromIndex(move: int):
    """
    Takes in the index of a move 0-63 and returns the cordanits 
    :param move: index of the move to be converted
    :return: row and column
    """
    row: int = (move // BOARD_SIZE) + 1  # 1-8
    col: str = chr(65+(move % BOARD_SIZE)) # A-H
    return row, col

def miniMax(Moves: list, currPlayer: int):
    """
    Implementation of the minimax algorithm with alpha beta pruning
    """
    index = -1
    return getCoordsFromIndex(index)


def heuristic(Board: npBoard, move: int):
    """
    Implementation of the heuristic function
    """
    score = -1
    return score


def search(gameBoard: npBoard, currPlayer: int):
    """
    Implementation of the search algorithm upon tree of moves
    """
    print(getLegalmoves(gameBoard, currPlayer))
    return -1

def out_of_bounds(row: int, col: int) -> bool:
        """
        Check if coordinates are out of bounds
        :param row: Integer row coordinate
        :param col: Integer column coordinate
        :return: True if out of bounds, false if on board
        """
        if (row < 0) or (row > 7) or (col < 0) or (col > 7):
            return True
        else:
            return False

main()  # run code
