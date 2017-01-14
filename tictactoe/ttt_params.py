

class Params():
    """NN params"""
    INPUTLEN = 18
    LABELLEN = 9
    LEARNING_RATE = 0.001
    HIDDENSIZE = 200
    SAVEEVERY = 100
    BATCHSIZE = 16

    """Reinforcement learning params"""
    GAMMA = 0.99
    UNSUP_SAMPLESIZE = 1024
    UNSUP_BATCHLENGTH = 1024

    TRAINTIME = 1000
    EPSILON_INIT = 1
    EPSILON_FINAL = 0.1

PARAMS = Params()
