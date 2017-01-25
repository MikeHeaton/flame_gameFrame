# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:29:07 2016

@author: Mike
"""
from config import PARAMS
import tensorflow as tf

# Collect customised classes from their locations
import importlib
GAMERULES = importlib.import_module(PARAMS.GAME_LOC).GameRules()

class NeuralNetwork():
    def __init__(self):
        """Required attachment points:
        self.state_placeholder          - placeholder for game state tuple;
        self.chosenmove_placeholder     - placeholder for the choice made in
                                            this particular instance,
                                            as a 1-hot tuple;
        self.scores_placeholder         - placeholder for the predictions of
                                            Q-scores at the current time step
                                            (calculated from the NN's
                                            predictions of Q-scores for each
                                            action at time t+1)
        self.reinforcement_train_op     - training operation which runs the
                                            loss calculation and backprop.
        TODO: saver? etc?
        """
        # State and move vector length can be inferred directly from
        # an example state, because they should be the same size for
        # each state.
        eg_state = GAMERULES.initial_state()
        self.state_vector_length = len(eg_state.as_tuple("X"))
        self.move_vector_length = len(GAMERULES.legal_moves(eg_state)[0].as_tuple())

        # Create the network according to the given specifications.
        [self.state_placeholder,
         self.chosenmove_placeholder,
         self.scores_placeholder]    = self.add_placeholders()

        self.score_predictions       = self.add_network(
                                                    self.state_placeholder)

        (self.reinforcement_train_op,
         self.total_loss)           = self.add_training(
                                                    self.score_predictions,
                                                    self.chosenmove_placeholder,
                                                    self.scores_placeholder)

    def add_placeholders(self):
        with tf.variable_scope("placeholders") as scope:
            state_placeholder      = tf.placeholder(tf.float32,
                                                    [None,
                                                     self.state_vector_length],
                                                    name="state_placeholder")
            chosenmove_placeholder = tf.placeholder(tf.float32,
                                                    [None,
                                                     self.move_vector_length],
                                                    name="chosenmove_placeholder")
            scores_placeholder     = tf.placeholder(tf.float32,
                                                    [None],
                                                    name="scores_placeholder")

        return state_placeholder, chosenmove_placeholder, scores_placeholder

    def add_network(self,   state_placeholder):
        # This method should return score_predictions, which is the Q-scores
        # for each move in the move vector.

        with tf.variable_scope("Layer") as scope:
            W1 = tf.get_variable("W1", [self.state_vector_length, PARAMS.hidden_layer_size])
            b1 = tf.get_variable("b1", [PARAMS.hidden_layer_size])
            layer_1 = tf.matmul(state_placeholder, W1) + b1
            layer_1 = tf.nn.relu(layer_1)

            # layer_1 = tf.nn.sigmoid(layer_1)

        with tf.variable_scope("LinClassifier") as scope:
            W2 = tf.get_variable("Voutput",
                                     [PARAMS.hidden_layer_size,
                                      self.move_vector_length])
            b2 = tf.get_variable("boutput",
                                     [self.move_vector_length])

            layer_2 = tf.matmul(layer_1,  W2) + b2

            score_predictions = layer_2

        return score_predictions

    def add_training(self,  score_predictions,
                            chosenmove_placeholder,
                            scores_placeholder):
        with tf.variable_scope("Loss") as scope:
            score_of_chosen_move = tf.reduce_sum(
                                        tf.mul(score_predictions,
                                              chosenmove_placeholder),
                                        axis=1)

            total_loss = tf.reduce_sum(tf.square(scores_placeholder -
                                                 score_of_chosen_move     ))

            rein_optimizer = tf.train.GradientDescentOptimizer(
                                                        PARAMS.learning_rate)
            rein_global_step = tf.Variable(0, name='rein_global_step',
                                           trainable=False)

            reinforcement_train_op = rein_optimizer.minimize(
                                                total_loss,
                                                global_step=rein_global_step)

        return reinforcement_train_op, total_loss
