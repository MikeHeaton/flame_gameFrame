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
        self.history = {}
        self.epsilon = epsilon

        self.sess = tf.Session()
        self.neuralnetwork = MODELCLASS(generate_mode=True)

        if PARAMS.learn_with_saved:
            saver = tf.train.Saver()
            saver.restore(self.sess, PARAMS.weights_location)
        else:
            self.sess.run(tf.initialize_all_variables())

    def play(self, state):
        """TODO: replace this copy/paste job with a direct call
        to the "play" method of the parent class."""
        def _internal_play(state):
            # Passes the state to the neural network as a tuple and receives
            # the estimated scores back in response.
            feed_dict = {self.neuralnetwork.state_placeholder: state.as_tuple(self.player)}
            self._estimated_scores = np.squeeze(self.sess.run(
                                            [self.neuralnetwork.score_predictions],
                                             feed_dict=feed_dict)[0])
            legalmoves = GAMERULES.legal_moves(state)

            bestmove = self._bestmove_from_scoresvector(self._estimated_scores,
                                                        legalmoves)
            return bestmove

        # Poll the NNPlayer method to get its opinion of the best move,
        # and record all of the associated scores (used for training).
        bestmove = _internal_play(state)
        scores_vector = self._estimated_scores
        self.history[state.as_tuple(self.player)] = scores_vector

        # With probability epsilon, overwrite the best move and replace
        # with a randomly selected move from all legal moves.
        if random.random() >= self.epsilon:
            return bestmove
        else:
            return random.choice(GAMERULES.legal_moves(state))

    def reset_history(self):
        self.history = {}

def runall():
    if PARAMS.train_NN_as == "X":
        trainingplayer = TrainingNNPlayer("X", epsilon=1)
        opposingplayer = MINMAX_PLAYER("O")
        traininggame = GAMEHANDLER(pX=trainingplayer, pO=opposingplayer, game_rules=GAMERULES)
    elif PARAMS.train_NN_as == "O":
        trainingplayer = TrainingNNPlayer("O", epsilon=1)
        opposingplayer = MINMAX_PLAYER("X")
        traininggame = GAMEHANDLER(pX=opposingplayer, pO=trainingplayer, game_rules=GAMERULES)

    for t in traininggame.run_game(first_player="X"):
        print(t["state"], t["player"], t["move"], t["score"], t["game_ended"])

runall()

"""____How do I do this again____
- Set up game and players

- Generate examples, turn them into game histories
(Note: make HISTORY objects? Probably! I like this nested structure for time-dimensioned data.
- Make batches out of game histories (consider how to do this with RNNs but maybe don't
                                        worry too much for now, this is the only bit which
                                        should need refactoring.)
- Crap... refactor so that we use variable batch sizes?
- Write a training step for a batch
- Put it all together in the outer function
"""




"""

def getbatch(G, epsilon):
    counter = 0
    history = []
    scorehistory = []
    wintype_history = []
    params = G.params
    while counter < params.UNSUP_BATCHLENGTH:
        # Play new games until we've recorded as many states as requested.

        iterator = G.gameiterator(newgame=True, epsilon=epsilon)
        gamehistory = []

        for gso in iterator:
            if gso["curplayer"] == NNPLAYER:
                gamehistory.append(gso)

        # After the game finishes, record the score and win type.
        # Remember, score is -1 if P0 wins, 1 if P1 wins, 0 if a draw
        # Need to maximise score so reverse it if NNPLAYER==0
        scorehistory.append(gso["score"] * (2*NNPLAYER-1))
        wintype_history.append(gso["wintype"])

        for t in range(len(gamehistory)-1, -1, -1):
            # Loop backwards over the game history (backwards is needed to get
            # rewards at time t+1)
            state = gamehistory[t]["inputvec"]
            action = gamehistory[t]["thisplay"]

            if t == len(gamehistory) - 1:
                gamehistory[t]["reward"] = scorehistory[-1]
            else:
                gamehistory[t]["reward"]  = gamehistory[t]["score"]

            # Label formula is:
            # Q(s,a) = r_(t+1) + GAMMA * max Q(s',a')
            # with s' the state at t+1
            # and a' ranging over possible actions.
            if t == len(gamehistory) - 1:
                gamehistory[t]["label"]  = gamehistory[t]["reward"]
            else:
                gamehistory[t]["label"]  = ( params.GAMMA *
                            np.max([gamehistory[t+1]["scoresestimate"][i]
                            for i in range(len(gamehistory[t+1]["scoresestimate"]))
                            if gamehistory[t+1]["legalmask"][i] == 1])
                           )
            label = gamehistory[t]["label"]
            history.append((state, action, label))
            counter += 1

    print(Counter(wintype_history))

    return history, scorehistory

def runepoch(m, sess, epsilon):
    # Run an epoch using m to play p0.

    print("Setting up model...")
    params = m.params
    if NNPLAYER == 1:
        p0 = perfectplayer.ttt_MinMaxPlayer(-1)
        p1 = ttt_players.ttt_NNPlayer(m, sess)
    elif NNPLAYER == 0:
        p1 = perfectplayer.ttt_MinMaxPlayer(1)
        p0 = ttt_players.ttt_NNPlayer(m, sess)
    else:
        print("NNPLAYER param not 1 or 0, not recognised.")
    G = ttt_game.Game(params, p0=p0, p1=p1)

    print("Generating play...")
    history, scorehistory = getbatch(G, epsilon)

    print("Sampling history...")
    sample = random.sample(history, params.UNSUP_SAMPLESIZE)
    minibatches = batchsample(sample, params)

    print("Training...")
    trainingloss = 0
    for states, actions, labels in minibatches:
        feeddict = {m.input_placeholder: states,
                    m.label_placeholder: actions,
                    m.score_placeholder: labels
                    }
        _, loss = sess.run([m.reinforcement_train_op, m.reinforcementloss], feed_dict = feeddict)
        trainingloss += loss

    print("Testing...")
    testscore = G.playgame()
    return trainingloss, np.mean(scorehistory), testscore

def batchsample(sample, params):
    # Takes a sample of states, and batches them into a form acceptable to the NN.
    # Samples are (statehistory, action, label)

    states_chunked = [[z[0] for z in sample[0+t:params.BATCHSIZE+t]] for t in range(0, len(sample), params.BATCHSIZE)]
    actions_chunked = [[z[1] for z in sample[0+t:params.BATCHSIZE+t]] for t in range(0, len(sample), params.BATCHSIZE)]
    rewards_chunked = [[z[2] for z in sample[0+t:params.BATCHSIZE+t]] for t in range(0, len(sample), params.BATCHSIZE)]
    return zip(states_chunked, actions_chunked, rewards_chunked)

def runtraining(params, usesaved=True):
    tf.reset_default_graph()
    with tf.Session() as sess:

        m = ttt_NN.ttt_NN(params)
        saver = tf.train.Saver()
        if usesaved:
            saver.restore(sess, './network_weights')
        else:
            init = tf.initialize_all_variables()
            sess.run(init)

        genscorehistory = []
        trainlosshistory = []
        testscorehistory = []
        epsilon = m.params.EPSILON_INIT

        for t in range(m.params.TRAINTIME):
            if epsilon > m.params.EPSILON_FINAL:
                epsilon -= 1/m.params.TRAINTIME
            print("--EPOCH {:d}--".format(t))

            trainloss, genscore, testscore = runepoch(m, sess, epsilon)
            genscorehistory.append(genscore)
            trainlosshistory.append(trainloss)
            testscorehistory.append(testscore)

            print("Epsilon: ", epsilon)
            print("Test average score: {:f}".format(genscore))
            print("Training loss: {:f}".format(trainloss))
            print("(Deterministic) game test score: {:d}".format(testscore))

            if t % params.SAVEEVERY == 0 and t > 0:
                saver.save(sess, './network_weights')
                print("SAVED")

        plt.rcParams["figure.figsize"] = (9,9)
        plt.figure(1)
        plt.subplot(311)
        plt.plot(trainlosshistory)
        plt.title("Train Loss History")
        plt.subplot(312)
        plt.plot(genscorehistory)
        plt.title("Generating (with Softmax) Scores History")
        plt.subplot(313)
        print(testscorehistory)
        plt.plot(testscorehistory)
        plt.title("Testing (best guess) Scores History")

    return trainlosshistory, genscorehistory, testscorehistory

if __name__ == '__main__':
    runtraining(Params(), usesaved=False)
"""
