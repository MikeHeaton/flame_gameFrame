from players import Player

class ttt_Manualplayer(Player):
    # Asks the player for input at each step
    def __init__(self):
        pass

    def play(self, state, legalmask, epsilon=0):
        ttt_game.printtttstate(state)
        vec = np.zeros([9])
        vec[input()] += 1
        return vec, vec

class ttt_NNPlayer(Player):
        # Implements an AI player which plays using a learned neural network.
    def __init__(self, model, sess):
        self.model = model
        self.sess = sess
        self.params = self.model.params

    def _runmodel(self, inputstate, legalmask):
        # Polls the NN model to get a good move and returns it.

        self.input = np.expand_dims(inputstate, axis=0)
        feeddict = {self.model.input_placeholder: self.input}
        scores = self.sess.run(self.model.logits, feed_dict=feeddict)[0]

        # Mask the scores to only look at _legal_ moves
        mask = (legalmask == 1)
        subset_idx = np.argmax(scores[mask])

        decision = np.zeros([self.model.params.LABELLEN])
        m = np.arange(scores.shape[0])[mask][subset_idx]
        decision[m] += 1
        return scores, decision

    def play(self, inputstate, legalmask, epsilon=0):
        # Poll self._runmodel to get the best move
        # Override it with probability epsilon
        # Note that training uses the overriden score, not the score of the
        # move returned by _runmodel.

        estscores, decision = self._runmodel(inputstate, legalmask)
        mask = (legalmask == 1)

        if np.random.rand() < epsilon:
            decision *= 0
            m = np.random.choice(np.arange(decision.shape[0])[mask])
            decision[m] += 1

        return estscores, decision
