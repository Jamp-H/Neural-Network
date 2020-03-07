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

    y_tild__train_vector = np.where(y_train==0, -1, y_train)
    y_tild__val_vector = np.where(y_val==0, -1, y_val)
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

        grad_matrix_list = []
        # loop over data points in subtrain data set
        index = 0
        for point in subtrain_X:

            # compute the gradients of V_mat/w_vec with respect to a
            # single observation in the subtrain set
            y_tild = y_tild__train_vector[index]
            h_list = forward_prop(point, V_mat)

            layer = 2
            while layer > 0:
                if layer == 2:
                    grad_a = (-y_tild) / 1 + np.exp(y_tild * h_list[layer-1])
                else:
                    grad_h = np.matmul(V_mat[layer], grad_a.reshape(10,1))

                    w_vec = h_list[layer]

                    grad_a = grad_h * w_vec * (1-w_vec)

                grad_val = np.matmul(grad_a.reshape(10,1), np.transpose(h_list[layer - 1].reshape(len(h_list[layer - 1]),1)))
                print(grad_val.shape)
                grad_matrix_list.append(grad_val)
                layer -= 1
            # update V.mat/w.vec by taking a step (scaled by step.size)
            # in the negative gradient direction
            for layer in range(0, len(grad_matrix_list)):
                grad_matrix_list[layer] -= step_size * grad_matrix_list[layer]
            index += 1

    # compute the logistic loss on the subtrain/validation sets
    # store value in loss_values (log loss formula log[1+exp(-y_tild * real_pred)])
    prediction_h_units_sub = forward_prop(np.transpose(subtrain_X), grad_matrix_list)
    prediction_h_units_val = forward_prop(np.transpose(validation), grad_matrix_list)

    train_loss = np.log(1+np.exp(-y_tild__train_vector*prediction_h_units_sub))
    val_loss = np.log(1+np.exp(-y_tild__val_vector*prediction_h_units_val))

    loss_values = [train_loss, val_loss]

    return (loss_values, V_mat, w_vec)
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

log_loss_csv(log_loss_data):
    with open("Neural_Net_Log_Loss_Train.csv", mode = 'w') as file:

        fieldnames = ['epochs', 'train error']
        writer = csv.DictWriter(file, fieldnames = fieldnames)

        writer.writeheader()
        epoch = 0
        for error in data[0][0].items():
            writer.writerow({'epoch': epoch,
                            'train error': error})
            epoch +=1

        with open("Neural_Net_Log_Loss_Val.csv", mode = 'w') as file:

            fieldnames = ['epochs', 'train error']
            writer = csv.DictWriter(file, fieldnames = fieldnames)

            writer.writeheader()
            epoch = 0
            for error in data[0][1].items():
                writer.writerow({'epoch': epoch,
                                'val error': error})
                epoch +=1



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
        h_list.append(a_vec)
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
    loss_data = n_net_one_split(X_mat, y_vec, 10, .05, 10, subtrain)

main()
