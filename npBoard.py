import random
import numpy as np
from enum import Enum
from Util import TerminalColor

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
    faster and more useable othello 
    """

    board = np.zeros(64)

    def __init__(self):
        self.set_piece_coords(5, 'D', -1)
        self.set_piece_coords(5, 'E', 1)
        self.set_piece_coords(4, 'D', 1)
        self.set_piece_coords(4, 'E', -1)

    def switchToFirstPlayer(self):
        self.board * -1

    def getBoard(self):
        return self.board

    def getCoordsFromIndex(move: int):
        """
        Takes in the index of a move 0-63 and returns the cordanits 
        :param move: index of the move to be converted
        :return: row and column
        """
        row: int = -1*((move // BOARD_SIZE) - 8) # 1-8
        col: str = chr(65+(move % BOARD_SIZE))  # A-H
        return row, col

    def _get_local_coords_from_index(index: int):
        """
        gets the local coordinate representation of an index (0-7, 0-7)
        :param move: index of move in question
        :return: row,column in 0-7,0-7 format
        """
        row: int = (index // BOARD_SIZE) # 0-7 range
        col: int = index % BOARD_SIZE # 0-7 range
        return row,col

    def set_piece_index(self, index: int, color: int):
        """
        Wrapper method for set piece so that you dont need to explictly convert indexes to coords
        :param index_move: index of the move to make
        :param color: Piececolor
        :return: False if illegal move, true otherwise
        """
        row, col = npBoard._get_local_coords_from_index(index)
        return self._set_piece(row,col,color)

    def set_piece_coords(self, _row: int, _col: str, color: int):
        """
        Set piece at the given coordinates to the given color (1 = us, -1 = them), takes in coords from the (int, str) format
        :param row: Row in the form 1-8
        :param col: Column in the form A-H
        :param color: PieceColor
        :return: False if this was an illegal move for some reason, otherwise True
        """
        row = _row - 1 # because row in this form is 1-8
        col = ord(_col) - ord('A')
        return self._set_piece(row,col,color)

    def _set_piece(self, row: int, col: str, color: int):
        """
        Set piece at the given coordinates to the given color (1 = us, -1 = them), generic form of set piece that abstracts the index args
        :param row: Row in the form 0-7
        :param col: Column in the form 0-7
        :param color: PieceColor
        :return: False if this was an illegal move for some reason, otherwise True
        """
        # Check if coordinates are out of bounds
        if npBoard._out_of_bounds(row, col):
            return False

        # Check if space is occupied
        if self.board[row * 8 + col] != 0:
            return False

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
            row_curr = row + row_off
            col_curr = col + col_off

            # List of pieces that could potentially be enveloped based on information known thus far
            potential_flip = []
            envelop = False
            while not npBoard._out_of_bounds(row_curr, col_curr):

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

                # Offset coordinates in direction
                row_curr += row_off
                col_curr += col_off

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

    def _get_player_positions(self, nextPiece: int):
        """
        Compiles a list of all player positions in local 0-7, 0-7 coordinates
        :param nextPiece: int of player in question
        :return: a list of positions in local 0-7, 0-7 coordinates
        """
        interestSpots = list()
        interestSpots = [i[0] for i, v in np.ndenumerate(self.getBoard()) if v == nextPiece]
        return [npBoard._get_local_coords_from_index(i) for i in interestSpots]

    def getLegalmoves(self, nextPiece: int):
        """
        Takes in a game board and the player who is about to move then returns a list of all legal moves
        :param Board: current board
        :param int: the color of the peice that is about to move (-1 for opponent 1 for us)
        :return: list of the index of all legal moves
        """
        # iterate through board, if selected piece, then propogate out
        # use a set because duplicate checking is O(1)
        legalMoves = set()
        interestSpots = self._get_player_positions(nextPiece)

        for piece in interestSpots:
            for step_row, step_col in [d.value for d in Direction]:
                # choose a search dir and compute a step for each iteration, iterate first and stop if we reach an edge
                row_ptr = piece[0] + step_row
                col_ptr = piece[1] + step_col
                readyForMove = False
                while not npBoard._out_of_bounds(row_ptr, col_ptr):
                    if(self.board[row_ptr * 8 + col_ptr] == 0):
                        # seen empty space, either break or add move
                        if(readyForMove):
                            legalMoves.add(row_ptr * 8 + col_ptr)
                        break
                    elif(self.board[row_ptr * 8 + col_ptr] == nextPiece):
                        # seen own piece, break propogation
                        break
                    else:
                        # seen enemy piece, next step could be valid move
                        readyForMove = True
                    row_ptr += step_row
                    col_ptr += step_col

        return list(legalMoves)

    def to_str(self, pot_moves: list(), player: int) -> str:
        """
        Convert this board to a pretty-printed string!
        :return: Pretty-printed string displaying board
        """
        out = ""
        for i in range(8):
            out += str(8 - i) + "  "
            for j in range(8):
                color = self._get_piece(i, j)
                temp = "0"
                if color == 1:
                    temp = TerminalColor.BLUE.value + "B" + TerminalColor.NRM.value
                elif color == -1:
                    temp = TerminalColor.RED.value + "R" + TerminalColor.NRM.value
                if (i*8 + j) in pot_moves:
                    if not temp == "0":
                        temp = TerminalColor.YELLOW.value + "X" + TerminalColor.NRM.value
                    else:
                        temp = TerminalColor.NEW.value + "X" + TerminalColor.NRM.value
                out += temp
                out += "  "
            out += "\n"
        out += "   A  B  C  D  E  F  G  H"
        return out


if __name__ == "__main__":
    print("GAMEBOARD TESTS")
    gameboard = npBoard()
    p = 1
    for i in range(100):
        p *= -1
        moves = gameboard.getLegalmoves(p)
        print("++++ player {} moves ++++".format(p))
        if moves == []:
            print("player {} has no moves left".format(p))
            break 
        chosen = random.choice(moves)
        print("Player {} has chosen this move {}".format(p,chosen ))
        print("all possible moves: {}".format([npBoard.getCoordsFromIndex(i) for i in moves]))
        print(gameboard.to_str(moves, p))
        print(" ")
        gameboard.set_piece_index(chosen,p)
