# flame_gameFrame
A flexible framework for teaching neural networks to play games. Forked off of
https://github.com/MikeHeaton/Deep-learning-TTT-web-demo but made flexible and adaptable.

# Creating a new game
To create a new game, the files which you need to edit are in the 'templates' folder.
Copy and rename 'templates'.
The implementation of the game lives in game.py. The classes to worry about are:

* GameState
  Contains the state of the game.
* GameMove
  Summarises a move of the game in a general form.
* GameRules
  Static methods for implementing the rules of the game.
