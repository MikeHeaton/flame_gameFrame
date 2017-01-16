
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

class NNPlayer(Player):
    def __init__(self, player):
        self.player = player
        self.sess = tf.Session()
        self.neuralnetwork = MODELCLASS()

    def _bestmove_from_scoresvector(self, estimated_scores, legalmoves):
        # Takes a set of estimated scores and a list of legal moves
        # in the position.
        # Convert these into a single best move and return it.
        raise NotImplementedError
        return MOVECLASS()

    def play(self, state):
        # Passes the state to the neural network as a tuple and receives
        # the estimated scores back in response.
        """TODO: CONVERT TO "X" or "O" STATE FOR PROPER FEEDING
        Maybe that should live in the state class?"""
        feed_dict = {self.model.state_placeholder: state.as_tuple()}
        estimated_scores = self.sess.run([self.state.score_predictions],
                                            feed_dict=feed_dict)
        legalmoves = GAME_RULES.legal_moves(state)

        bestmove = self._bestmove_from_scoresvector(estimated_scores,
                                                    legalmoves)
        return bestmove
