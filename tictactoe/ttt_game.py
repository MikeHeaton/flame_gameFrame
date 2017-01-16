#-*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:40:45 2016

@author: Mike
"""

import numpy as np

class TTTMove():
    """ Describes a logical move in the game. """

    def as_tuple(self):
        # Express the state of the game as a 1-d list of integers.
        # This is what is fed into the neural network.
        val = 3 * self.Y + self.X
        tup = np.zeros([9])
        tup[val] = 1
        return tup

    def __init__(self, move):
        # state is a 1-d representation of the state, suitable
        # for feeding into a neural network.
        self.Y = move[0]
        self.X = move[1]

    def __str__(self):
        # Return a string representing the state in a human-readable format.
        return "({}, {})".format(self.Y, self.X)

class TTTBoard():
    """ Encompasses the physical state of the game.
        Doesn't include any consideration of the rules, that is
        handled by the GameRules class below."""

    def as_tuple(self, player):
        # Express the state of the game as a 1-d list of integers.
        # This is what is fed into the neural network.
        # 'player' determines for which player ("X" or "O") the board
        # is from the perspective of.
        playerint = {"X": -1, "O": 1}[player]
        flattened = np.reshape(self.state, [9])
        tupled = np.squeeze(np.concatenate(([flattened == playerint],
                                            [flattened == -playerint]),
                                           axis=1)).astype(int)
        return tuple(tupled)

    def __init__(self, state):
        # state is a 1-d representation of the state, suitable
        # for feeding into a neural network.
        self.state = state

    def __str__(self):
        # Return a string representing the state in a human-readable format.
        outputstring = "___\n"
        for Y in range(3):
            for X in range(3):
                if self.state[Y, X] == -1:
                    outputstring += "X"
                elif self.state[Y, X] == 1:
                    outputstring += "O"
                else:
                    outputstring += "."
            outputstring += "\n"
        outputstring += "---"
        return outputstring

class TTTGame():
    """------Managing the rules of Tic Tac Toe------"""

    @staticmethod
    def initial_state():
        # Return the base state that the game is in at time 0.
        return TTTBoard(np.zeros([3,3]))

    @staticmethod
    def legal_moves(state):
        # Returns all the moves which you want to expose an agent to
        # making in a given state.
        return [TTTMove((Y, X)) for (Y, X) in [(0,0), (0,1), (0,2),
                                    (1,0), (1,1), (1,2),
                                    (2,0), (2,1), (2,2)]
                            if state.state[Y, X] == 0]

    @staticmethod
    def update_state(state, player, decision):
        # Applies a decision vector from player player ('X'/'O') to update the state.
        # Return the new updated state, or None if an illegal move was made.
        # For this game, the decision vector is in [0,1,2]x[0,1,2].
        if player == "X":
            playerinteger = -1
        elif player == "O":
            playerinteger = 1
        else:
            raise(ValueError, "Player code {} not recognised.".format(player))

        if state.state[decision.Y, decision.X] != 0:
            print("Illegal move!")
            print(state)
            print(decision)
            return None
        else:
            newstate = state.state.copy()
            newstate[decision.Y, decision.X] = playerinteger
            return TTTBoard(newstate)

    @staticmethod
    def eval_state(state):
        # Evaluate a game state.
        # Return an int for the score: -1, 0, 1 for X win, draw, O win,
        # or return None for 'game continues'.
        has_won = TTTGame._haswon(state)
        if has_won:
            return has_won
        elif TTTGame._hasdrawn(state):
            return 0
        else:
            return None

    @staticmethod
    def _haswon(state):
        # Checks board for winners. Returns -1/1 if there's a winner or 0 else.
        for p in [-1, 1]:
            # Check rows / columns
            for i in range(3):
                if (np.sum(state.state[i,:])  == p * 3 or
                    np.sum(state.state[:,i]) == p * 3) :
                    return p

            # Check diagonals
            if (    np.sum([state.state[i,i] for i in range(3)]) == p * 3 or
                    np.sum([state.state[i,2-i] for i in range(3)]) == p * 3 ):
                return p

        return False

    @staticmethod
    def _hasdrawn(state):
        # If there's an empty square on the board, we're not done yet.
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                if state.state[i,j] == 0:
                    return False
        return True
