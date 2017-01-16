from tictactoe.ttt_game import *
from tictactoe.ttt_neural import *
from tictactoe.ttt_neuralplayer import *
from players import *
from game_handlers import *
from config import GAME_RULES
import tensorflow as tf

if __name__ == "__main__":

    testplayerX = MinMaxPlayer("X")
    testNNplayerO = NNPlayer("O")

    testgame = Game(pX=testplayerX, pO=testNNplayerO, game_rules=GAME_RULES)
    print(testgame.play_game())

    """

    testboard = TTTBoard.from_tuple(input_state)
    print(testboard)

    with tf.Session() as sess:
        testmodel = TTTModel(generate_mode=True)
        sess.run(tf.initialize_all_variables())
        feed_dict = {testmodel.state_placeholder : input_state,
                    testmodel.chosenmove_placeholder : move_vector,
                    testmodel.scores_placeholder : scores_vector,
                    }
        x = sess.run([testmodel.score_predictions], feed_dict=feed_dict)"""
