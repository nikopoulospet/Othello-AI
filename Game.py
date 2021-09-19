"""
This file comprises a portion of the source code of the Othello referee implementation for the WPI course 'CS 4341:
Introduction to Artificial Intelligence' running A term of the 2021-2022 academic year.

File:   Game.py
Author: Dyllan Cole <dcole@wpi.edu>
Date:   14 September 2021
"""

from enum import Enum

from Board import Board, PieceColor
from Util import TerminalColor


class EndCondition(Enum):
    """
    Enum representing all the possible end conditions of the game
    """
    INVALID = "Invalid move!"
    OOO = "Out-of-order move!"
    TIME_OUT = "Time out!"
    VALID = "The winning player has more discs on the board!"
    TIE = "Match TIED!"


class Game:
    """
    Class representing a game of Othello
    """
    player_one = None
    player_two = None
    _player_curr = None
    board = None
    game_over = False

    def __init__(self, player_one: str, player_two: str):
        """
        Initialize Othello game
        :param player_one: Group name of player one
        :param player_two: Group name of player two
        """
        self.player_one = player_one
        self.player_two = player_two

        # Initialize this game's board
        self.board = Board()

    def get_next_player(self) -> str:
        """
        Get next player
        :return: Player whose turn is next
        """
        if (self._player_curr is None) or (self._player_curr == self.player_two):
            self._player_curr = self.player_one
        else:
            self._player_curr = self.player_two

        return self._player_curr

    def get_opponent(self, player: str) -> str:
        """
        Get opponent of given player
        :param player: Player to get opponent for
        :return: Group name of opponent of player
        """
        if player == self.player_one:
            return self.player_two
        else:
            return self.player_one

    def get_color(self, player: str) -> PieceColor:
        """
        Get PieceColor for the given player to use
        :param player: Player to check color for
        :return: PieceColor of given player
        """
        if player == self.player_one:
            return PieceColor.BLUE
        else:
            return PieceColor.ORANGE

    def has_legal_move(self, player) -> bool:
        """
        Check if given player has any valid moves that they can take
        :param player: Player to check moves for
        :return: True if there is a valid move to be taken, otherwise false
        """
        return self.board.has_valid_move(self.get_color(player))

    def end(self, reason: EndCondition, winner: str = None):
        """
        Handle printing out information about end of game and creating end_game file
        :param reason: Reason for game end
        :param winner: Optional winning player of game
        :return: None
        """
        msg = None
        if reason == EndCondition.VALID:
            counts = self.board.get_counts()
            if counts[self.get_color(self.player_one)] == counts[self.get_color(self.player_two)]:
                msg = "Match TIED!"
            elif counts[self.get_color(self.player_one)] > counts[self.get_color(self.player_two)]:
                winner = self.player_one
            else:
                winner = self.player_two
        if msg is None:
            loser = self.get_opponent(winner)
            msg = "{w} WINS! {l} LOSES! {r}".format(w=winner, l=loser, r=reason.value)
        with open("end_game", "w") as fp:
            fp.write(msg)

        color = TerminalColor.GREEN.value if EndCondition == EndCondition.VALID else TerminalColor.RED.value
        print(color + "Game Over: {m}".format(m=msg) + TerminalColor.NRM.value)
        self.game_over = True
