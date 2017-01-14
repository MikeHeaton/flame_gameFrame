
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:23:54 2016

@author: Mike
"""

from config import GAME_RULES
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

    def _minmax(self, state, player, depth):
        # We don't allow for memoisation recall for depth=0, because we want
        # to choose at random from all of the equally successful moves.
        if (state.as_tuple(), self.player) in self.state_dictionary and depth > 0:
            return self.state_dictionary[(state.as_tuple(), self.player)]

        # If we're at a leaf, return the score for it.
        evaluation = GAME_RULES.eval_state(state)
        if evaluation is not None:
            self.state_dictionary[(state.as_tuple(), player)] = (evaluation, None)
            return evaluation, None

        # Else, scroll through the legal moves.
        else:
            playerinteger = {"X": -1, "O": 1}[player]
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

            self.state_dictionary[(state.as_tuple(), player)] = (bestoutcome, bestmove)
            return bestoutcome, bestmove
