# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:19:31 2016

@author: Mike
"""
from config import PARAMS
import tensorflow as tf
import numpy as np
import random

# Collect customised classes from their locations
import importlib
GAMERULES = importlib.import_module(PARAMS.GAME_LOC).GameRules
GAMEHANDLER = importlib.import_module(PARAMS.GAMEHANDLERS_LOC).Game
MODELCLASS = importlib.import_module(PARAMS.NEURALMODEL_LOC).NeuralNetwork
NEURAL_PLAYER = importlib.import_module(PARAMS.NEURALPLAYER_LOC).NNPlayer
MINMAX_PLAYER = importlib.import_module(PARAMS.MINMAXPLAYER_LOC).MinMaxPlayer

class TrainingNNPlayer(NEURAL_PLAYER):
    # Wrapper class for NEURAL_PLAYER which records the histories of its
    # confidences when asked to make a decision, and exposes the neural
    # network directly so that it can be trained.

    # Resetting the history is of course necessary whenever the weights of
    # the network change, ie whenever the model is trained, ie each epoch.
    def __init__(self, player, epsilon):
        self.player = player
        self.history = []
        self.epsilon = epsilon

        self.sess = tf.Session()
        self.neuralnetwork = MODELCLASS()

        self.saver = tf.train.Saver()
        if PARAMS.learn_with_saved:
            self.saver.restore(self.sess, PARAMS.weights_location)
        else:
            self.sess.run(tf.global_variables_initializer())

    def play(self, state):

        # Poll the NNPlayer method to get its opinion of the best move,
        # and record all of the associated scores (used for training).
        bestmove = super(TrainingNNPlayer,self).play(state)
        scores_vector = self.estimated_scores
        self.history.append(scores_vector)

        # With probability epsilon, overwrite the best move and replace
        # with a randomly selected move from all legal moves.
        if random.random() >= self.epsilon:
            return bestmove
        else:
            return random.choice(GAMERULES.legal_moves(state))

    def reset_history(self):
        self.history = []

    def get_history(self):
        return self.history

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def create_gameandplayers():
    if PARAMS.train_NN_as == "X":
        trainingplayer = TrainingNNPlayer("X", epsilon=1)
        opposingplayer = MINMAX_PLAYER("O")
        traininggame = GAMEHANDLER(pX=trainingplayer, pO=opposingplayer,
                                    record_neural="X")
    elif PARAMS.train_NN_as == "O":
        trainingplayer = TrainingNNPlayer("O", epsilon=1)
        opposingplayer = MINMAX_PLAYER("X")
        traininggame = GAMEHANDLER(pX=opposingplayer, pO=trainingplayer,
                                    record_neural="O")

    return traininggame, trainingplayer

def generate_histories(traininggame, trainingplayer, N):
    # Takes a game and a number N of samples to generate.
    # Plays the game until N (or more) samples have been seen.
    # Returns the history of each game from nnplayer's perspective.

    totalstates = 0
    allgames = []
    gameoutcomes = {-1:0, 0:0, 1:1}
    targetscore = {"X":-1, "O":1}

    # Generate samples until we have enough from the NNplayer's POV.
    while totalstates < N:
        thisgame = traininggame.run_game(first_player="X")

        from_nn_pov = thisgame.filtered(PARAMS.train_NN_as)
        predictions_history = trainingplayer.get_history()
        allgames.append(list(zip(from_nn_pov, predictions_history)))

        totalstates += len(from_nn_pov)
        gameoutcomes[thisgame.outcome] += 1

    print("Finished generating, scores: {}".format(gameoutcomes))
    return allgames

def run_eval(traininggame, trainingplayer, N):
    # Takes a game and a number N of samples to generate.
    # Plays the game until N (or more) samples have been seen.
    # Returns the history of each game from nnplayer's perspective.
    currentepsilon = trainingplayer.epsilon

    trainingplayer.set_epsilon(0)

    gameoutcomes = {-1:0, 0:0, 1:1}
    targetscore = {"X":-1, "O":1}

    # Play the game N times to get results
    for _ in range(N):
        thisgame = traininggame.run_game(first_player="X")
        gameoutcomes[thisgame.outcome] += 1

    trainingplayer.set_epsilon(currentepsilon)

    return gameoutcomes

def create_batches(gamehistories, trainingplayer):
    # Takes a list of game histories and the NNPlayer used to generate them.
    # Extracts the trainingplayer's score predictions for each time step
    # and uses them to create training targets for each sample in the history.
    # Batches them into numpy arrays and yields out the arrays.
    """TODO: if we want to use RNNs, this will need to be refactored."""
    targetvalues_list = []
    state_list = []
    chosenmove_list = []
    for game in gamehistories:
        for t in range(len(game)):
            # Create training examples from each turn.
            # A training example is the state; a one-hot of the chosen move;
            # and a value representing the target prediction value, equal to:
            # Q(s,a) = max Q(s', a')
            # with s' being the state at the NEXT time step.

            # For the last turn in the game, there is no next value
            # so just use the score.

            # Note that turn.score is NOT relative to the player, so "X" winning
            # gives a score of -1. This isn't what we want because states
            # are fed in relative to the player's own position, so we need to
            # adjust for this.
            player = trainingplayer.player
            playerinteger = {"X":-1, "O":1}[player]

            # Add the immediate reward (0 except for at game end)
            targetvalue = game[t][0].score * playerinteger

            if t < len(game) - 1:
                nexttime_predictions = game[t+1][1]
                nexttime_legalmoves  = game[t+1][0].legalmovestuple
                # print(game[t+1][0].state)
                # print(nexttime_predictions, nexttime_legalmoves)
                # Add the max Q-score among _legal_ moves at time t+1.
                targetvalue += (PARAMS.gamma *
                                max([x[0] for x in zip(nexttime_predictions,
                                                       nexttime_legalmoves)
                                     if x[1]]))
                # print(targetvalue)

            #print(game[t][0].state, game[t][0].move, targetvalue)

            targetvalues_list.append(targetvalue)

            state_list.append(game[t][0].state.as_tuple(player))
            chosenmove_list.append(game[t][0].move.as_tuple())
    #for i in list(zip(state_list, chosenmove_list, targetvalues_list)):
    #    print(i[0], i[1], i[2])

    def batch_generator(state_list, chosenmove_list, targetvalues_list):
        t = 0
        while t < len(state_list):
            state_batch = np.vstack(state_list[t: t+PARAMS.batch_size])
            chosenmove_batch = np.vstack(chosenmove_list[t: t+PARAMS.batch_size])
            targetvalues_batch = np.squeeze(np.vstack(targetvalues_list[t: t+PARAMS.batch_size]), axis=1)
            yield {
                    "states": state_batch,
                    "moves" : chosenmove_batch,
                    "scores": targetvalues_batch
                  }

            t += PARAMS.batch_size

    return batch_generator(state_list, chosenmove_list, targetvalues_list)

def train_step(minibatch, trainingplayer):
    # Feed a minibatch into the neural network of the trainingplayer.
    # Train it using gradient descent.
    NN = trainingplayer.neuralnetwork

    feed_dict = {
                 NN.state_placeholder      : minibatch["states"],
                 NN.chosenmove_placeholder : minibatch["moves"],
                 NN.scores_placeholder     : minibatch["scores"]
                }

    _, loss = trainingplayer.sess.run(
                        [NN.reinforcement_train_op, NN.total_loss],
                        feed_dict=feed_dict)

    return loss

def runepoch(traininggame, trainingplayer, N):
    trainingplayer.reset_history()

    print("Playing games...")
    history = generate_histories(traininggame, trainingplayer, N)

    print("Training network...")
    total_loss = 0
    for minibatch in create_batches(history, trainingplayer):
        total_loss += train_step(minibatch, trainingplayer)

    return total_loss

def runall():
    print("Setting up game...")
    traininggame, trainingplayer = create_gameandplayers()

    epsilon = 1
    for n in range(PARAMS.num_epochs):
        print("---EPOCH {}---".format(n))

        epsilon = max(PARAMS.min_epsilon, epsilon - 1/PARAMS.num_epochs)
        trainingplayer.set_epsilon(epsilon)
        print("Epsilon = {}".format(np.round(epsilon, 4)))
        total_loss = runepoch(traininggame, trainingplayer, PARAMS.epoch_length)
        print("Epoch finished, total loss: {}".format(total_loss))

        if (n+1) % PARAMS.save_every == 0:
            trainingplayer.saver.save(trainingplayer.sess,
                                      PARAMS.weights_location)

            print("WEIGHTS SAVED")

        if (n+1) % PARAMS.eval_every == 0:
            results = run_eval(traininggame, trainingplayer, PARAMS.eval_length)
            print("EVALUATION COMPLETE. Scores: {}".format(results))

if __name__ == "__main__":
    runall()

"""THINK: when we override the choice with gamma, are we using the original choice
or the new choice to train with? I think the new choice but double-check."""
