# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:29:07 2016

@author: Mike
"""

from config import PARAMS, GAME_RULES
import tensorflow as tf

class NeuralNetwork():
    def __init__(self, generate_mode=False):
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
        # Batch size depends on whether we're training,
        # or generating live predictions.
        self.generate_mode = generate_mode

        # State and move vector length can be inferred directly from
        # an example state, because they should be the same size for
        # each state.
        eg_state = GAME_RULES.initial_state()
        self.state_vector_length = len(eg_state.as_tuple())
        self.move_vector_length = len(GAME_RULES.legal_moves(eg_state)[0])

        # Create the network according to the given specifications.
        [self.state_placeholder,
         self.chosenmove_placeholder,
         self.scores_placeholder]    = self.add_placeholders()

        # Reshape the placeholders if necessary
        batch_size = PARAMS.batch_size if not self.generate_mode else 1
        [reshaped_state_placeholder,
        reshaped_chosenmove_placeholder,
        reshaped_scores_placeholder]    = [tf.reshape(x, [batch_size, -1])
                                           for x in
                                           [self.state_placeholder,
                                            self.chosenmove_placeholder,
                                            self.scores_placeholder]
                                           ]

        self.score_predictions       = self.add_network(
                                                    reshaped_state_placeholder)

        self.reinforcement_train_op  = self.add_training(
                                                    self.score_predictions,
                                                    reshaped_chosenmove_placeholder,
                                                    reshaped_scores_placeholder)

    def add_placeholders(   self):
        with tf.variable_scope("placeholders") as scope:
            if not self.generate_mode:
                state_placeholder      = tf.placeholder(tf.float32,
                                                        [PARAMS.batch_size,
                                                         self.state_vector_length],
                                                        name="state_placeholder")
                chosenmove_placeholder = tf.placeholder(tf.float32,
                                                        [PARAMS.batch_size,
                                                         self.move_vector_length],
                                                        name="chosenmove_placeholder")
                scores_placeholder     = tf.placeholder(tf.float32,
                                                        [PARAMS.batch_size],
                                                        name="scores_placeholder")
            else:
                state_placeholder      = tf.placeholder(tf.float32,
                                                        [self.state_vector_length])
                chosenmove_placeholder = tf.placeholder(tf.float32,
                                                        [self.move_vector_length])
                scores_placeholder     = tf.placeholder(tf.float32,
                                                        [])

        return state_placeholder, chosenmove_placeholder, scores_placeholder


def add_network(self,   state_placeholder):
    # This method should return score_predictions, which is the Q-scores
    # for each move in the move vector.

    with tf.variable_scope("Layer") as scope:
        raise NotImplementedError
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

    return reinforcement_train_op
