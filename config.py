from tictactoe import ttt_game

class PARAMS:
    """-----NN parameters------"""
    hidden_layer_size = 200
    batch_size = 32
    learning_rate = 0.001
    weights_location = './network_weights'

    """---Playing parameters---"""
    play_with_saved = False
    minmax_save_states = True


GAME_RULES = ttt_game.TTTGame()
BOARD_CLASS = ttt_game.TTTBoard
MOVE_CLASS = ttt_game.TTTMove
