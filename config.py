class PARAMS:
    modules_location = "tictactoe."

    """-------NN parameters---------"""
    hidden_layer_size = 200
    batch_size = 32
    learning_rate = 0.001
    weights_location = './network_weights'

    """-----Learning parameters-----"""
    learn_with_saved = False
    train_NN_as = "X"

    """-----Playing parameters------"""
    play_with_saved = False
    minmax_save_states = True

    """-----Pointers to modules and classes-----"""
    # These locations are used for dynamic import
    # using importlib.import_module().
    GAME_LOC = modules_location + "game"
    GAMEHANDLERS_LOC = "game_handlers"
    NEURALMODEL_LOC = modules_location + "neuralmodel"

    MINMAXPLAYER_LOC = "minmax_player"
    MANUALPLAYER_LOC = modules_location + "manual_player"
    NEURALPLAYER_LOC = modules_location + "neural_player"
