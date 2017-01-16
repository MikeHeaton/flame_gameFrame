
class Game():
    """------Methods to handle playing a full game------"""
    def __init__(self, pX, pO, game_rules, initial_state=None):
        self.players = {"X": pX, "O": pO}
        self.game_rules = game_rules

        if not initial_state:
            initial_state = self.game_rules.initial_state()
        self.state = initial_state

        self.game_history = []

    def run_game(self, first_player="X"):
        current_player = first_player
        other_player = {"X":"O", "O":"X"}
        current_state = self.game_rules.initial_state()
        gamehistory = []
        while True:
            turn_stats, current_state = self.run_move(current_state, current_player)
            gamehistory.append(turn_stats)

            if turn_stats["game_ended"]:
                break

            current_player = other_player[current_player]

        return gamehistory

    def run_move(self, state, player):
        # Poll player for a move, apply it to state.
        # Return details about the results of the turn, and the new state.
        """TODO: edit newstate_evaluation to add support
           for games which give score over time"""
        response  = self.players[player].play(state)
        newstate = self.game_rules.update_state(state,
                                                 player,
                                                 response)
        newstate_evaluation = self.game_rules.eval_state(newstate)

        this_turn = {
                    "state"      : state,
                    "player"     : player,
                    "move"       : response,
                    "score"      : (newstate_evaluation if newstate_evaluation is not None else 0),
                    "game_ended" : (newstate_evaluation is not None)
                    }

        return this_turn, newstate

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
