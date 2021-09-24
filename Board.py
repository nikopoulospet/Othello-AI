"""
This file comprises a portion of the source code of the Othello referee implementation for the WPI course 'CS 4341:
Introduction to Artificial Intelligence' running A term of the 2021-2022 academic year.

File:   Board.py
Author: Dyllan Cole <dcole@wpi.edu>
Date:   14 September 2021
"""

from enum import Enum

from Util import TerminalColor


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


class PieceColor(Enum):
    """
    Enum of possible piece colors, or none for an empty spot, on an Othello board
    """
    NONE = "*"
    BLUE = TerminalColor.BLUE.value + "B" + TerminalColor.NRM.value
    ORANGE = TerminalColor.YELLOW.value + "O" + TerminalColor.NRM.value


def interpret_coords(row: int, col: str) -> (int, int):
    """
    Convert coordinates from system specified in the project description to internal form.
    :param row: Row in the form 1-8
    :param col: Column in the form A-H
    :return: Tuple of (row: 0-7, col: 0-7)
    """
    return 8 - row, ord(col) - ord('A')


def transform_coords(row: int, col: int) -> (int, str):
    """
    Convert coordinates from internal form to system specified in the project description.
    :param row: Row in the form 0-7
    :param col: Column in the form 0-7
    :return: Tuple of (row: 1-8, col: A-h)
    """
    return 8 - row, chr(ord('A') + col)


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


class Board:
    """
    Class representing an Othello board
    """
    board = [PieceColor.NONE] * 64

    def __init__(self):
        """
        Initialize Othello board
        """
        # Setup initial board state
        self.set_piece(5, 'D', PieceColor.BLUE)
        self.set_piece(5, 'E', PieceColor.ORANGE)
        self.set_piece(4, 'D', PieceColor.ORANGE)
        self.set_piece(4, 'E', PieceColor.BLUE)

    def _get_piece(self, row: int, col: int) -> PieceColor:
        """
        Get piece at given coordinates on Othello board
        :param row: Row in the form 0-7
        :param col: Column in the form 0-7
        :return: PieceColor at (row, col) on board
        """
        return self.board[row * 8 + col]

    def get_piece(self, row: int, col: str) -> PieceColor:
        """
        Get piece at given coordinates on Othello board
        :param row: Row in the form 1-8
        :param col: Column in the form A-H
        :return: PieceColor at (row, col) on board
        """
        # Convert coordinates to internal form
        coords = interpret_coords(row, col)
        return self._get_piece(coords[0], coords[1])

    def _get_enveloped_pieces(self, row: int, col: int, color: PieceColor) -> [[int, int]]:
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
            while not out_of_bounds(row_curr + row_off, col_curr + col_off):
                # Offset coordinates in direction
                row_curr += row_off
                col_curr += col_off

                # Check if piece could be enveloped
                color_curr = self._get_piece(row_curr, col_curr)
                if color_curr == PieceColor.NONE:
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

    def _set_piece(self, row: int, col: int, color: PieceColor) -> bool:
        """
        Set piece at the given coordinates to the given color
        :param row: Row in the form 0-7
        :param col: Column in the form 0-7
        :param color: PieceColor
        :return: False if this was an illegal move for some reason, otherwise True
        """

        # Check if coordinates are out of bounds
        if out_of_bounds(row, col):
            return False

        # Check if space is occupied
        if self.board[row * 8 + col] != PieceColor.NONE:
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

    def set_piece(self, row: int, col: str, color: PieceColor) -> bool:
        """
        Set piece at the given coordinates to the given color
        :param row: Row in the form 1-8
        :param col: Column in the form
        :param color: PieceColor
        :return: False if move was illegal, otherwise True
        """

        # Convert coordinates to internal representation
        coords = interpret_coords(row, col)
        return self._set_piece(coords[0], coords[1], color)

    def is_full(self) -> bool:
        """
        Check if the board is completely full
        :return: True if the board is completely full, otherwise False
        """
        return self.board.count(PieceColor.NONE) == 0

    def get_counts(self) -> {}:
        """
        Get the counts of each PieceColor currently on the board
        :return: Dict of {PieceColor: Count, ...}
        """

        # Track running count
        blue = 0
        orange = 0

        # Count pieces of each color on board
        for color in self.board:
            if color == PieceColor.BLUE:
                blue += 1
            elif color == PieceColor.ORANGE:
                orange += 1

        return {PieceColor.BLUE: blue, PieceColor.ORANGE: orange}

    def has_valid_move(self, color: PieceColor) -> bool:
        """
        Check if there exists a valid move for the given color
        :param color: PieceColor
        :return: True if there is a valid move, otherwise False
        """
        for row in range(8):
            for col in range(8):
                piece = self._get_piece(row, col)
                if (piece == PieceColor.NONE) and len(self._get_enveloped_pieces(row, col, color)) > 0:
                    return True
        return False

    def __str__(self) -> str:
        """
        Convert this board to a pretty-printed string!
        :return: Pretty-printed string displaying board
        """
        out = ""
        for i in range(8):
            out += str(8 - i) + "  "
            for j in range(8):
                color = self._get_piece(i, j)
                out += color.value
                out += "  "
            out += "\n"
        out += "   A  B  C  D  E  F  G  H"
        return out
