class PARAMS:
    modules_location = "connectiv."

    """-------NN parameters---------"""
    hidden_layer_size = 500
    batch_size = 32
    learning_rate = 0.0001
    weights_location = './network_weights/weights'

    """-----Learning parameters-----"""
    learn_with_saved = False
    train_NN_as = "O"
    epoch_length = 1000
    num_epochs = 10000
    save_every = 100
    min_epsilon = 0.1
    gamma = 0.99
    minmax_maxdepth = 1

    eval_every = 50
    eval_length = 1000

    """-----Playing parameters------"""
    play_with_saved = True
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
