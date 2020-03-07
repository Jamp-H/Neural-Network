###############################################################################
#
# AUTHOR(S): Joshua Holguin
# DESCRIPTION: program that will inplement stochastic gradient descent algorithm
# for a one layer neural network
# VERSION: 0.0.1v
#
###############################################################################
import numpy as np
from sklearn.preprocessing import scale
from scipy.stats import norm
from random import randint

# Function: n_net_one_split
# INPUT ARGS:
# X_mat          : train inputs/feature matrix, n_observations x n_features
# y_vec          : train outputs/label vector, n_observations x 1
# max_epochs     : int scalar > 1
# step_size      : double scalar > 0
# n_hidden_units : int scalar>1, number of hidden units
# is_subtrain    : logical vector, size n_observations
# Return: loss_values, V_mat, w_vec
def n_net_one_split(X_mat, y_vec, max_epochs, step_size, n_hidden_units, is_subtrain):
    # initialize design of layered neural network
    design = (X_mat.shape[1], n_hidden_units, 1)

    # divide X_mat and y_vec into train, validation (60% 40% respectively)
    X_train, X_val = split_train_val(X_mat)
    y_train, y_val = split_train_val(y_vec)

    validation = (is_subtrain == 1)

    # get subtrain_X and subtrain_y of of training data and is_subtrain
    subtrain_X = X_train[is_subtrain, :]

    # initilize V_mat list (weight matrix n_features x n_hidden_units)
        # used to predict hidden units given input

    V_mat = []
    # loop though design of layers
    # initilize each matrix; add to V_mat
    for i in range(0,len(design) - 1):
        # calculate dimensions and num of entries of new matrix
        new_mat_row = design[i + 1]
        new_mat_col = design[i]
        num_of_entries = new_mat_row * new_mat_col

        # create matrix
        new_mat = np.random.randn(new_mat_row, new_mat_col)
        new_mat /= 10
        # append matrix to list of matricies (V_mat)
        V_mat.append(new_mat)

    # loop over epochs (k=1 to max_epochs)
        # update the parameters using the gradients with respect to each
        # subtrain observation
    for epoch in range(0,max_epochs):
        i = 0
        # loop over data points in subtrain data set
        for point in subtrain_X:
            # compute the gradients of V_mat/w_vec with respect to a
            # single observation in the subtrain set
            h_list = forward_prop(point, V_mat)
            # update V.mat/w.vec by taking a step (scaled by step.size)
            # in the negative gradient direction
            print(h_list)
        # compute the logistic loss on the subtrain/validation sets
        # store value in loss_values (log loss formula log[1+exp(-y_tild * real_pred)])

    # return outputs (loss_values, V_mat, w_vec)

# Function: convert_data_to_matrix
# INPUT ARGS:
#   file_name : the csv file that we will be pulling our matrix data from
# Return: data_matrix_full
def convert_data_to_matrix(file_name):
    data_matrix_full = np.genfromtxt( file_name, delimiter = " " )
    return data_matrix_full

# Function: split matrix
# INPUT ARGS:
#   X : matrix to be split
# Return: train, validation matriceis
def split_train_val(X):
    train, validation = np.split( X, [int(.6 * len(X))])

    return (train, validation)

# Function: forward_prop
# INPUT ARGS:
# in_mat          : input row from matrix (1 x features)
# list_of_weights : List of maricies containing weights
# Return: hidden layer vector
def forward_prop(in_row, list_of_weights):
    h_list = []
    h_list.append(in_row)
    for layer_i in range(0, len(list_of_weights) - 1):
        weight = list_of_weights[layer_i]
        hidden_layer = h_list[layer_i]
        a_vec = np.matmul(weight, hidden_layer)
    if layer_i == len(list_of_weights) - 1:
        h_list.append(a_vec)
    else:
        a_vec = 1/(1 + np.exp(-a_vec))
    return h_list
# Function: main
# INPUT ARGS:
#   none
# Return: none
def main():
    np.random.seed(1)

    file_name = "spam.data"
    # Get data into matrix form
    X_mat_full = convert_data_to_matrix(file_name)

    X_mat_full_col_len = X_mat_full.shape[1]
    # split data into a matrix and vector of pred values
    X_mat = X_mat_full[:,:-1]
    y_vec = X_mat_full[:,X_mat_full_col_len-1]

    # Scale matrix for use
    X_sc = scale(X_mat)

    subtrain = np.random.randint(1,5,X_mat_full_col_len-1)


    # dummy data so not use to actually test
    n_net_one_split(X_mat, y_vec, 10, .05, 10, subtrain)

main()
