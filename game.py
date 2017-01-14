#-*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:40:45 2016

@author: Mike
"""

class GameState():
    """ Encompasses the physical state of the game.
        Doesn't include any consideration of the rules, that is
        handled by the GameRules class below."""
    def __init__(self, state):
        # 'state' is any convenient structure representing the state of the
        # game. You can choose to expose it to the GameRules class if
        # convenient, but it is NOT used by the learning algorithm.
        self.state = state

    def from_tuple(state_tuple):
        # Create the game state from its tuple representation.
        # The tuple representation is a 1-d representation of the state,
        # suitable for feeding into a neural network.
        raise NotImplementedError

    def as_tuple(self):
        # Express the state of the game as a 1-d list of integers.
        # This is what is fed into the neural network.
        raise NotImplementedError

    def __str__(self):
        # Return a string representing the state in a human-readable format.
        raise NotImplementedError

class GameRules():
    """Manages the rules of the game.
        Contains static methods determining how the game state
        is to be interpreted and evolved."""
    @staticmethod
    def initial_state(state):
        # Return the base state that the game is in at time 0.
        raise NotImplementedError

    @staticmethod
    def legal_moves(state):
        # Returns all the moves which you want to expose an agent to
        # making in a given state.
        raise NotImplementedError

    @staticmethod
    def eval_state(state):
        # Evaluate a game state.
        # Return an int for the score: -1, 0, 1 for X win, draw, O win,
        # or return None for 'game continues'.
        raise NotImplementedError

    @staticmethod
    def update_state(state, player, decision):
        # Applies a decision vector from player player ('X'/'O') to update the state.
        raise NotImplementedError
