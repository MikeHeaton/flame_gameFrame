
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:23:54 2016

@author: Mike
"""

from config import PARAMS
import random
import numpy as np

# Collect customised classes from their locations
import importlib
GAMERULES = importlib.import_module(PARAMS.GAME_LOC).GameRules()

class MinMaxPlayer():
    # Implements a generalised MinMax algorithm.
    """TODO: a/b pruning"""
    """TODO: max depth"""
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

    def _heuristic_state(self, state):
        """TODO: Make this customisable"""
        if len(GAMERULES.legal_moves(state)) == 0:
            return 0, None
        else:
            return 0, random.choice(GAMERULES.legal_moves(state))

    def _minmax(self, state, player, depth):
        # We don't allow for memoisation recall for depth=0, because we want
        # to choose at random from all of the equally successful moves.
        if (PARAMS.minmax_save_states and
            state.as_tuple(player) in self.state_dictionary and
            depth > 0):
            bestoutcome, bestmove = self._fetchfrom_dict(state, player)

        else:
            # If we're at a leaf, return the result for it. This is in absolutes:
            # -1 for X win, 1 for O win, 0 for draw, None for no score.
            evaluation = GAMERULES.eval_state(state)
            if evaluation is not None:
                bestoutcome, bestmove = evaluation, None

            else:
                # If we've exceeded the max depth, evaluate the state and return a
                # heuristic.
                if depth > PARAMS.minmax_maxdepth:
                    bestoutcome, bestmove = self._heuristic_state(state)

                else:
                    # Else, scroll through the legal moves.
                    playerinteger = {"X": -1, "O": 1}[player]
                    bestoutcome = None
                    bestmove = None
                    possiblemoves = GAMERULES.legal_moves(state)
                    random.shuffle(possiblemoves)

                    for move in possiblemoves:
                        # If we have a better move than current, remember it.
                        next_state = GAMERULES.update_state(state, player, move)
                        otherplayer = {"X":"O", "O":"X"}[player]
                        outcome, _ = self._minmax(next_state, otherplayer, depth+1)

                        if (bestoutcome is None or
                                np.abs(playerinteger - outcome) <
                                np.abs(playerinteger - bestoutcome)):
                            bestoutcome = outcome
                            bestmove = move

            # We don't want to save the state if we just looked it up;
            # in all other situations we do.
            if PARAMS.minmax_save_states:
                self._addto_dict(state, player, bestoutcome, bestmove)

        return bestoutcome, bestmove
