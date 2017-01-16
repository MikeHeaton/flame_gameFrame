# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:23:54 2016
@author: Mike
"""
from config import PARAMS
import tensorflow as tf
import numpy as np

# Collect customised classes from their locations
import importlib
GAMERULES = importlib.import_module(PARAMS.GAME_LOC).GameRules()
MOVECLASS = importlib.import_module(PARAMS.GAME_LOC).GameMove
MODELCLASS = importlib.import_module(PARAMS.NEURALMODEL_LOC).NeuralNetwork

class NNPlayer():
    def __init__(self, player):
        self.player = player
        self.sess = tf.Session()

        # Add the name of the neural network class above here.
        self.neuralnetwork = MODELCLASS(generate_mode=True)

        if PARAMS.play_with_saved:
            saver = tf.train.Saver()
            saver.restore(self.sess, PARAMS.weights_location)
        else:
            self.sess.run(tf.initialize_all_variables())

    def _bestmove_from_scoresvector(self, estimated_scores, legalmoves):
        # Takes a 1d array of estimated scores and a list of legal moves
        # in the position.
        # Convert these into a single best move and return it.
        legalmovetups = [(i//3, i%3) for i in range(9)
                                if i in map(lambda m: 3*m.Y+m.X, legalmoves)]
        bestmovetuple = max(legalmovetups,
                             key=lambda tup: estimated_scores[3*tup[0]+tup[1]])
        return MOVECLASS(bestmovetuple)

    def play(self, state):
        # Passes the state to the neural network as a tuple and receives
        # the estimated scores back in response.
        feed_dict = {self.neuralnetwork.state_placeholder: state.as_tuple(self.player)}
        self.estimated_scores = np.squeeze(self.sess.run(
                                        [self.neuralnetwork.score_predictions],
                                         feed_dict=feed_dict)[0])
        legalmoves = GAMERULES.legal_moves(state)

        bestmove = self._bestmove_from_scoresvector(self.estimated_scores,
                                                    legalmoves)
        return bestmove
