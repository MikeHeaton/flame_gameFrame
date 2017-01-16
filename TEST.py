from tictactoe.ttt_game import *
from tictactoe.ttt_neural import *
from tictactoe.ttt_neuralplayer import *
from players import *
from game_handlers import *
from config import GAME_RULES
import tensorflow as tf

if __name__ == "__main__":

    testplayerX = MinMaxPlayer("X")
    testNNplayerO = MinMaxPlayer("O")

    testgame = Game(pX=testplayerX, pO=testNNplayerO, game_rules=GAME_RULES)
    [print(i["state"], i["player"], i["move"]) for i in testgame.run_game()]
