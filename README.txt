Author: Dyllan Cole <dcole@wpi.edu>
Date:   14 September 2021

Description:
    This program serves as the referee for a game of Othello played between two other programs. It will detect whether
moves are invalid / valid, and score the game accordingly, as per the rules laid out in the Project 2 description. It is
entirely possible that the referee contains bugs; if you so much as suspect that the referee is not handling something
properly, please let the course staff know on the appropriate discussion board on Canvas.
    Please note that this program is only a REFEREE. It will never make any moves itself. If you want to test your AI,
you will either need to run it against itself, or some other test program. To start with, it might be handy to have a
test player that will perform a random legal move.

Prerequisites:
    * Python 3.5+

Usage:
    python Referee.py <GROUP 1> <GROUP 2>