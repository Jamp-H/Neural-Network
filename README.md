# Single Layered Neural Network

Link to our source code for the single layered Neural Network function: https://github.com/Jamp-H/Neural-Network/blob/master/neural_network.py

Our function for the signle layered neural network takes in a set of parameters in order to compute a overall list of gradiants that are based off of the number of epochs entered in. this list of gradients is then used to calculate our logistic loss for the train and validation data we split the original data into.

## How to run this function
This function can be run by inputting the following information into the function call:
  X_mat          : train inputs/feature matrix, n_observations x n_features
  y_vec          : train outputs/label vector, n_observations x 1
  max_epochs     : int scalar > 1
  step_size      : double scalar > 0
  n_hidden_units : int scalar>1, number of hidden units
  is_subtrain    : logical vector, size n_observations
