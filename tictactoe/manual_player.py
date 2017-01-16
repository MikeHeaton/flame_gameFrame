# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 11:23:54 2016

@author: Mike
"""
from config import PARAMS

# Collect customised classes from their locations
import importlib
GAMEMOVE = importlib.import_module(PARAMS.GAME_LOC).GameMove()

class ManualPlayer():
    # Asks the user for input at each step. Used for demo purposes.
    def __init__(self, player):
        self.player = player

    def play(self, state):
        # Displays the state to ßthe user and asks for input.
        # Processes the input and returns it as the response.
        print(state)
        decision_string = input()

        # Process the decision_string and return a move object
        # which the game rules class will recognise.
        decision = decision_string
        raise NotImplementedError

        return GAMEMOVE(decision)
