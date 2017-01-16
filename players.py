
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:23:54 2016

@author: Mike
"""

from config import GAME_RULES, BOARD_CLASS, MOVE_CLASS, PARAMS
import random
import math

class Player():
    # Base class for other types of players. Don't edit.
    def __init__(self, player):
        self.player = player
        raise NotImplementedError

    def play(self, state):
        raise NotImplementedError


class ManualPlayer(Player):
    # Asks the user for input at each step. Used for demo purposes.
    def __init__(self, player):
        self.player = player

    def play(self, state):
        # Displays the state to the user and asks for input.
        # Processes the input and returns it as the response.
        print(state)
        decision_string = input()

        # Process the decision_string and return a decision vector
        # which the game rules class will recognise.
        raise NotImplementedError


class MinMaxPlayer(Player):
    # Implements a generalised MinMax algorithm.
    """TODO: a/b pruning"""
    def __init__(self, player):
        self.player = player
        self.state_dictionary = {}

    def play(self, state):
        """TODO: a/b pruning"""
        bestoutcome, bestmove = self._minmax(state, self.player, 0)
        return bestmove

    def _fetchfrom_dict(self, state, player):
        # Fetch the best outcome for the current player as recorded in the
        # self.state_dictionary.
        # The score is recorded relatively, as 1 for a win and -1 for a loss,
        # IE assuming player is "O". Adjust for this and return
        # the actual best score and move.
        playerinteger = {"X": -1, "O": 1}[player]

        reloutcome, bestmove =  self.state_dictionary[state.as_tuple(player)]
        return reloutcome * playerinteger, bestmove

    def _addto_dict(self, state, player, outcome, move):
        # Fetch the best outcome for the current player as recorded in the
        # self.state_dictionary.
        # The score is recorded as 1 for a win and -1 for a loss,
        # IE assuming player is "O". Adjust for this and return
        # the actual best score and move.
        playerinteger = {"X": -1, "O": 1}[player]
        self.state_dictionary[state.as_tuple(player)] = (outcome * playerinteger, move)
        return 0

    def _minmax(self, state, player, depth):
        # We don't allow for memoisation recall for depth=0, because we want
        # to choose at random from all of the equally successful moves.
        playerinteger = {"X": -1, "O": 1}[player]

        if (PARAMS.minmax_save_states and
            state.as_tuple(player) in self.state_dictionary and
            depth > 0):
            bestoutcome, bestmove = self._fetchfrom_dict(state, player)

        else:
            # If we're at a leaf, return the result for it. This is in absolutes:
            # -1 for X win, 1 for O win, 0 for draw, None for no score.
            evaluation = GAME_RULES.eval_state(state)
            if evaluation is not None:
                bestoutcome, bestmove = evaluation, None

            # Else, scroll through the legal moves.
            else:
                bestoutcome = None
                bestmove = None
                possiblemoves = GAME_RULES.legal_moves(state)
                random.shuffle(possiblemoves)

                for move in possiblemoves:
                    # If we have a better move than current, remember it.
                    next_state = GAME_RULES.update_state(state, player, move)
                    otherplayer = {"X":"O", "O":"X"}[player]
                    outcome, _ = self._minmax(next_state, otherplayer, depth+1)

                    if (bestoutcome is None or
                            math.fabs(playerinteger - outcome) <
                            math.fabs(playerinteger - bestoutcome)):
                        bestoutcome = outcome
                        bestmove = move

            if PARAMS.minmax_save_states:
                self._addto_dict(state, player, bestoutcome, bestmove)

        return bestoutcome, bestmove
