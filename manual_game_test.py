from config import PARAMS

# Collect customised classes from their locations
import importlib
GAMERULES = importlib.import_module(PARAMS.GAME_LOC).GameRules
GAMEHANDLER = importlib.import_module(PARAMS.GAMEHANDLERS_LOC).Game
MODELCLASS = importlib.import_module(PARAMS.NEURALMODEL_LOC).NeuralNetwork
NEURAL_PLAYER = importlib.import_module(PARAMS.NEURALPLAYER_LOC).NNPlayer
MINMAX_PLAYER = importlib.import_module(PARAMS.MINMAXPLAYER_LOC).MinMaxPlayer
MANUAL_PLAYER = importlib.import_module(PARAMS.MANUALPLAYER_LOC).ManualPlayer

pX = MANUAL_PLAYER("X")
pO = MANUAL_PLAYER("O")

test = GAMEHANDLER(pX, pO)

test.run_game()
