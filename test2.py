from Game import Game
import os.path


def main():
    gameOver = False
    gameboard = Game('first','second')
    myPlace = 'second'
    theirPlace = 'first'
    while(not gameOver):

        #if game is over break
        if(os.path.isfile('end_game.txt')):
            print('GG')
            gameOver = True
            continue

        #if not my turn break
        if(not os.path.isfile(__file__ + '.go')):
            print('passing')
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
            myPlace = 'first'
            theirPlace = 'second'
        else: # if there is a move that exists from the oponet do it
            # Tokenize move
            tokens = line.split()
            col = tokens[1]
            row = tokens[2]
            gameboard.board.set_piece(int(row), col, gameboard.get_color(theirPlace))

        # update internal board

        # play move 
        row = '' #1-8
        col = '' #A-H

        # move making logic

        #update model
        gameboard.board.set_piece(int(row), col, gameboard.get_color(myPlace))
        
        #send move
        file = open('move_file', 'w')
        file.write(__file__ + " " + col + " " + row)
        file.close()
main()