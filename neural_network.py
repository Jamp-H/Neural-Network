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
    # initialize architecture of layered neural network
    architecture = (X_col.shape[1], n_hidden_units, 1)

    # divide X_mat and y_vec into train, validation (60% 40% respectively)
    X_train, X_val = split_train_val(X_mat)
    y_train, y_val = split_train_val(y_vec)

    # initilize V_mat (weight matrix n_features x n_hidden_units)
        # used to predict hidden units given input
    V_mat
    for i in range(0,len(architecture)):
        
    # initilize w_vec (weight vector n_hidden_units size)
        # used to predict output given hidden units


    # loop over epochs (k=1 to max_epochs)
        # update the parameters using the gradients with respect to each
        # subtrain observation

        # loop over data points in subtrain data set

            # compute the gradients of V-mat/w-vec with respect to a a
            # single observation in the subtrain set

            # update V.mat/w.vec by taking a step (scaled by step.size)
            # in the negative gradient direction

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
    train, validation = np.split( X, int(.6 * len(X)))

    return {"train": train, "val": validation}


# Function: main
# INPUT ARGS:
#   none
# Return: none
def main():
    file_name = "spam.data"
    # Get data into matrix form
    X_mat_full = convert_data_to_matrix(file_name)

    X_mat_full_col_len = X_mat_full.shape[1]
    # split data into a matrix and vector of pred values
    X_mat = X_mat_full[:,:-1]
    y_vec = X_mat_full[:,X_mat_full_col_len-1]

    # Scale matrix for use
    X_sc = scale(X_mat)

main()
