
from config import PARAMS

# Collect customised classes from their locations
import importlib
GAMERULES = importlib.import_module(PARAMS.GAME_LOC).GameRules

class Game():
    """------Methods to handle playing a full game------"""
    def __init__(self, pX, pO, initial_state=None, record_neural=None):
        self.players = {"X": pX, "O": pO}
        self.record_neural = record_neural

        if not initial_state:
            initial_state = GAMERULES.initial_state()
        self.state = initial_state

        self.game_history = []

    def run_game(self, first_player="X", verbose=False):
        current_player = first_player
        other_player = {"X":"O", "O":"X"}
        current_state = GAMERULES.initial_state()
        gamehistory = []
        while True:
            turn = self.run_move(current_state, current_player)
            gamehistory.append(turn)
            if verbose:
                print(turn.state, turn.player, turn.move, turn.score, turn.neuralscores)
            if turn.game_ended:
                break

            current_player = other_player[current_player]
            current_state = turn.next_state

        return GameHistory(gamehistory)

    def run_move(self, state, player):
        # Poll player for a move, apply it to state.
        # Return details about the results of the turn, and the new state.
        """TODO: edit newstate_evaluation to add support
           for games which give score over time"""
        response  = self.players[player].play(state)

        neuralscores = (self.players[player].estimated_scores
                        if self.record_neural is not None and
                            (player == self.record_neural or
                             player in self.record_neural    )
                        else None)

        this_turn = Turn(state, player, response, neuralscores)

        return this_turn

class Turn():
    # Contains useful information about a turn.
    def __init__(self, state, player, move, neuralscores=None):
        self.state = state
        self.player = player
        # print(state)
        # print([i.as_tuple() for i in GAMERULES.legal_moves(state)])

        self.legalmovestuple = list((bool(sum(x)) for x in zip(*[i.as_tuple() for i in GAMERULES.legal_moves(state)])))
        # print(self.legalmovestuple)
        self.move = move
        self.neuralscores = neuralscores

        """if neuralscores is not None:
            print(self.state)
            print(self.legalmovestuple, self.neuralscores, [x[0] for x in zip(self.neuralscores,
                                                self.legalmovestuple)
                                                        if x[1]])
                                                        """
        self.next_state = GAMERULES.update_state(state,
                                                 player,
                                                 move)
        newstate_evaluation = GAMERULES.eval_state(self.next_state)
        self.score = (newstate_evaluation if newstate_evaluation is not None else 0)
        self.game_ended = (newstate_evaluation is not None)

class GameHistory(list):
    # Class to hold a list of Turn objects.
    def __init__(self, turns):
        super(GameHistory,self).__init__(turns)
        if len(turns) > 0:
            self.outcome = turns[-1].score
            self.winner = {-1: "X", 1: "O", 0: None}[self.outcome]
        else:
            self.outcome = None
            self.winner = None

        # Correct the scores: if the opponent won the game on their last move,
        # that won't be recorded in the player's last move (because they hadn't
        # lost yet), so this needs to be corrected.
        """NOTE if a player can take multiple subsequent turns this will fail,
        OR if scores update dynamically through the game. Would need to generalise it."""
        self[-2].score = self.outcome

    def filtered(self, player):
        # Return the history of only those moves played by player, as a list.
        return list(filter(lambda x: x.player == player, self))

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
