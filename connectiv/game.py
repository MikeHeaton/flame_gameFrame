#-*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:40:45 2016

@author: Mike
"""
import numpy as np

class GameMove():
    """ Describes a logical move in the game. """

    def as_tuple(self):
        # Express the move as a 1-d tuple of integers.
        # Almost certainly this should be a 1-hot vector.
        # This is what is fed into the neural network.
        tup = np.zeros([9])
        tup[self.pos] = 1
        return tuple(tup)

    def __init__(self, move):
        # move is a representation of a move. as_tuple is suitable
        # for feeding into a neural network.
        self.pos = move

    def __str__(self):
        # Return a string representing the state in a human-readable format.
        return str(self.pos)

class GameState():
    """ Encompasses the physical state of the game.
        Doesn't include any consideration of the rules, that is
        handled by the GameRules class below."""
    def __init__(self, state):
        # 'state' is any convenient structure representing the state of the
        # game. You can choose to expose it to the GameRules class if
        # convenient, but it is NOT used by the learning algorithm.

        # State will be a 4x7 matrix.
        self.state = state

    def as_tuple(self, player):
        # Express the state of the game as an array of integers.
        # This is what is fed into the neural network.
        # 'player' determines for which player ("X" or "O") the board
        # is from the perspective of.
        return self.state

    def __str__(self):
        # Return a string representing the state in a human-readable format.
        return str(self.state)

class GameRules():
    """Manages the rules of the game.
        Contains static methods determining how the game state
        is to be interpreted and evolved."""
    @staticmethod
    def initial_state():
        # Return the base state that the game is in at time 0.
        return GameState(np.zeros([6,7]))

    @staticmethod
    def legal_moves(state):
        # Returns all the moves which you want to expose an agent to
        # making in a given state.
        return [GameMove(i) for i in list(range(7)) if state.state[0, i] == 0]

    @staticmethod
    def eval_state(state):
        # Evaluate a game state.
        # Return an int for the score: -1, 0, 1 for X win, draw, O win,
        # or return None for 'game continues'.
        for p in [-1, 1]:
            # Check verticals
            for col in state.state.T:
                for i in range(3):
                    if sum(col[i:i+4]) == p*4:
                        return p

            # Check horizontals
            for row in state.state:
                for i in range(4):
                    if sum(row[i:i+4]) == p*4:
                        return p

            # Check diagonals
            for x in range(4):
                for y in range(3):
                    if sum([state.state[y+i, x+i] for i in range(4)]) == p*4:
                        return p

                    if sum([state.state[5-(y+i), x+i] for i in range(4)]) == p*4:
                        return p

        # Check draw
        if len(GameRules.legal_moves(state)) == 0:
            return 0

        # If no winner, keep going
        return None

    @staticmethod
    def update_state(state, player, decision):
        # Applies a decision vector from player player ('X'/'O') to update the state.
        playerint = {"X": -1, "O": 1}[player]
        y = 0
        while y < 6 and state.state[y, decision.pos] == 0:
            y += 1

        newstate = state.state.copy()
        newstate[y-1, decision.pos] = playerint
        return GameState(newstate)
