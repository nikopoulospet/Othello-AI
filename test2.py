from Board import Board, PieceColor
from Game import Game
import os.path

#NOTE: Blue is first place

def main():
    gameOver = False
    gameboard = Game('first','second')
    myColor = PieceColor.ORANGE
    theirColor = PieceColor.BLUE
    while(not gameOver):

        #if game is over break
        if(os.path.isfile('end_game.txt')):
            print('GG EZ') #TODO remove for improved runtime
            gameOver = True
            continue

        #if not my turn break
        if(not os.path.isfile(__file__ + '.go')):
            print('passing') #TODO remove for improved runtime
            #maybe add move scanning here to save time?
            #or start caculating possable furture moves
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
            print("Let me go first") #TODO remove for improved runtime
            theirColor = PieceColor.ORANGE
            myColor = PieceColor.BLUE
        else: # if there is a move that exists from the oponet do it
            # Tokenize move
            tokens = line.split()
            col = tokens[1]
            row = tokens[2]
            # update internal board
            gameboard.board.set_piece(int(row), col, theirColor)

        print(gameboard.board) #TODO remove for improved runtime

        # Find all legal moves
        legalMoves = getLegalmoves(gameboard.board, myColor)

        # move making logic
        bestMove:int = -1
        for move in legalMoves:
            #TODO Write
            pass

        # convert index to move
        row, col = getCordsFromIndex(bestMove)
       
        #update model
        gameboard.board.set_piece(int(row), col, myColor)

        #send move
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
    legalMoves = list()
    # TODO: Write
    return legalMoves
    
def getCordsFromIndex(move:int):
    """
    Takes in the index of a move 0-63 and returns the cordanits 
    :param move: index of the move to be converted
    :return: row and colume
    """
    row:int = -1 #1-8
    col:str = '' #A-H
    # TODO: Write
    return row, col

main() #run code