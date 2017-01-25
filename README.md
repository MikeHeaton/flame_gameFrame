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

neuralmodel.py contains the neural model. It's written in tensorflow and can be
made how you wish. Only add_network should need to be written, as the interface
of the model with the outside world should be fixed.

The other files have short methods to update to manage their exact interface with the game,
these should be easy. Only ad

Check out the example game files for details!

# Training a game

unsupervisedlearn.py is the script to train a neural network.

config.py contains training parameters including number of iterations, batch sizes, etc.

If you've written the template files correctly, this should all run. For first training
make sure that learn_with_saved is False. For subsequent runs (retraining the same params)
set learn_with_saved=True.
