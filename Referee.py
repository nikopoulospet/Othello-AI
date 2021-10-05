"""
This file comprises a portion of the source code of the Othello referee implementation for the WPI course 'CS 4341:
Introduction to Artificial Intelligence' running A term of the 2021-2022 academic year.

File:   Referee.py
Author: Dyllan Cole <dcole@wpi.edu>
Date:   14 September 2021
"""


import argparse
import os
import random
import re
import sys
import time
from os import listdir
from os.path import isfile, join

from Game import Game, EndCondition
from Util import TerminalColor


def clean():
    """
    Delete files maintained by Referee
    :return: None
    """
    patterns = [
        re.compile("move_file"),
        re.compile("(:?.*).go"),
        re.compile("end_game")
    ]

    files = [f for f in listdir("./") if isfile(join("./", f))]
    for file in files:
        for pattern in patterns:
            if pattern.match(file):
                os.remove(file)


def main():
    """
    Main Referee function
    :return: None
    """

    # Read in arguments from command line
    parser = argparse.ArgumentParser(description="Referee a game of Othello between two programs")
    parser.add_argument("player_one", type=str, help="Group name of player one")
    parser.add_argument("player_two", type=str, help="Group name of player two")
    parser.add_argument("--no_color", default=False, action='store_true')
    args = parser.parse_args(sys.argv[1:])

    # Wipe out coloring if flag is on
    if args.no_color:
        TerminalColor.RED = ""
        TerminalColor.NRM = ""
        TerminalColor.BLUE = ""
        TerminalColor.YELLOW = ""
        TerminalColor.GREEN = ""

    # Select order randomly
    p1 = args.player_one
    p2 = args.player_two
    if random.choice([True, False]):
        # Swap p1 and p2
        p3 = p1
        p1 = p2
        p2 = p3

    # Clean any pre-existing files
    clean()

    # Create game
    game = Game(p1, p2)

    # Display initial board
    print("Initial Board:\n{b}\n".format(b=game.board))

    # Create empty move_file
    open("move_file", "w").close()

    # Play game until the board is full or there are no more legal moves
    while not game.board.is_full() and (game.has_legal_move(game.player_one) or game.has_legal_move(game.player_two)):
        player = game.get_next_player()

        # Remove old go file if there is one
        old = "{p}.go".format(p=game.get_opponent(player))
        if os.path.exists(old):
            os.remove(old)

        # Get last modified time of move file and signal next player to go
        mtime = os.path.getmtime("move_file")
        open("{p}.go".format(p=player), "w").close()

        # Sleep for 10 seconds checking if move_file has been modified every 50 milliseconds
        modified = False
        for i in range(200):
            time.sleep(0.05)

            if os.path.getmtime("move_file") > mtime:
                modified = True
                break

        if modified:
            with open("move_file", "r") as fp:
                # Get last non-empty line from file
                line = ""
                for next_line in fp.readlines():
                    if next_line.isspace():
                        break
                    else:
                        line = next_line

                # Tokenize move
                tokens = line.split()
                group_name = tokens[0]
                col = tokens[1]
                row = tokens[2]

                # Verify that move is from expected player
                if group_name == game.get_opponent(player):
                    game.end(EndCondition.OOO, player)
                    break

                # Check if move is valid
                print("Move: {m}".format(m=line.rstrip()))
                if (col == 'P') and (game.has_legal_move(player)):
                    game.end(EndCondition.INVALID, game.get_opponent(player))
                    break
                elif (col == 'P') or game.board.set_piece(int(row), col, game.get_color(player)):
                    print("Board:\n{b}\n".format(b=game.board))
                else:
                    game.end(EndCondition.INVALID, game.get_opponent(player))
                    break
        else:
            # Player didn't move in time!
            game.end(EndCondition.TIME_OUT, game.get_opponent(player))
            break

    # End game if not already over
    if not game.game_over:
        game.end(EndCondition.VALID)

    # Print final state of board
    print("Final Board:\n{b}".format(b=game.board))


if __name__ == "__main__":
    main()
