
class GameHandler():
    """------Methods to handle playing a full game------"""
    def __init__(self, pX, pO, game_rules, initial_state=None):
        self.players = {"X": pX, "O": pO}
        self.game_rules = game_rules

        if not initial_state:
            initial_state = self.game_rules.initial_state()
        self.state = initial_state

        self.game_history = []

    def game_iterator(self, first_player="X"):
        current_player = first_player
        other_player = {"X":"O", "O":"X"}

        while True:
            response   = self.players[current_player].play(self.state)

            self.game_history.append(this_turn)
            self.new_state = self.game_rules.update_state(  self.state,
                                                            player,
                                                            response)
            state_evaluation = game_rules.eval_state(self.state)

            this_turn = {
                        "state" : self.state,
                        "player": current_player,
                        "move"  : response,
                        "score" : (state_evaluation if state_evaluation is not None else 0)
                        }
            yield this_turn

            if game_rules.eval_state(self.state) is not None:
                break

            current_player = other_player[current_player]

    def play_game(self, first_player="X"):
        return [turn for turn in self.game_iterator(first_player=first_player)]

'''
class NetworkGame(Game):
    """Simpler class with a method for taking in a state and a move, updating
    the state, askingan NN for its move, updating the state, and returning."""
    def __init__(self, model, sess):
        self.aiPlayer = ttt_players.ttt_NNPlayer(model, sess)
        self.state = None
        self.session = sess

    def playmove(self, inputvector):
        # Interpret inputvector as the state
        self.ravelinputvector(inputvector)

        # Get AI player move
        legalmask = self.getacceptabilitymask()
        estscores, aimove = self.aiPlayer.play(inputvector, legalmask, epsilon=0)

        # Update state for move
        self.updatestate(1,aimove)

        # Return new state
        return self.getstate(0)

    def ravelinputvector(self, inputvector):
        self.state = np.zeros([3,3,2])
        xvector = inputvector[:9]
        ovector = inputvector[9:]

        self.state[:,:,0] = np.reshape(xvector, [3,3])
        self.state[:,:,1] = np.reshape(ovector, [3,3])'''
