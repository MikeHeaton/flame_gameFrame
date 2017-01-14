from tictactoe.ttt_game import TTTBoard
from players import MinMaxPlayer
from config import GAME_RULES

if __name__ == "__main__":
    testplayer = MinMaxPlayer("X")

    testboard = TTTBoard.from_tuple([1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
    print(testboard)
    print(testplayer.play(testboard))
