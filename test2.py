from Board import Board, PieceColor, Direction
from Game import Game
import os.path

import random
import numpy as np
# NOTE: Blue is first place

BOARD_SIZE = 8


def main():
    gameOver = False
    gameboard = Game('first', 'second')
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
            gameboard.board.set_piece(int(row), col, theirColor)

        print(gameboard.board)  # TODO remove for improved runtime
        # Find all legal moves
        legalMoves = getLegalmoves(gameboard.board, myColor)

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


def getLegalmoves(Board: Board, nextPiece: PieceColor):
    """
    Takes in a game board and the player who is about to move then returns a list of all legal moves
    :param Board: current board
    :param nextPiece: the color of the peice that is about to move
    :return: list of the index of all legal moves
    """
    # iterate through board, if selected piece, then propogate out
    # use a set because duplicate checking is O(1)
    legalMoves = set()
    interestSpots = [i for i, v in Board.board if v == nextPiece.name]
    for piece in interestSpots:
        for direction in Direction:
            # choose a search dir and compute a step for each iteration, iterate first and stop if we reach an edge
            step = direction.value[0] + (direction.value[1]*BOARD_SIZE)
            searchIndex = piece
            readyForMove = False
            while 1:
                searchIndex += step
                if(Board.board[searchIndex] == PieceColor.NONE):
                    # seen empty space, either break or add move
                    if(readyForMove):
                        legalMoves.add(searchIndex)
                    break
                elif(Board.board[searchIndex] == nextPiece):
                    # seen own piece, break propogation
                    break
                else:
                    # seen enemy piece, next step could be valid move
                    readyForMove = True
                if(onBoardEdge(searchIndex)):
                    # if a piece is on the edge we know there is no more spaces to search
                    break
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
    col: str = chr(65+(move % BOARD_SIZE))  # A-H
    return row, col


def miniMax(gameboard: Board):
    """
    Implementation of the minimax algorithm with alpha beta pruning
    :param myMoves are the next possible legal moves for our player
    :param opponentMoves are the next possible legal moves for our opponent with know heuristic
    :param currPlayer is the current player based on the color
    :return the optimal move
    """

    # 1 is our piece, -1 is opponent piece, 0 is empty spot

    # get legal moves after
    # set_piece to do each move
    # get legal moves again for opponent moves, set_piece for all of those and run heuristic to get board state value
    # return that heuristic value then run minimax aglo on that

    index = -1

    return getCoordsFromIndex(index)


def heuristic(currBoard: Board):
    """
    Implementation of the heuristic function
    """
    score = -1
    return score


def search(gameBoard: Board, currPlayer: PieceColor):
    """
    Implementation of the search algorithm upon tree of moves
    """
    print(getLegalmoves(gameBoard, currPlayer))
    return -1


main()  # run code
