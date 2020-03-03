###############################################################################
#
# AUTHOR(S): Joshua Holguin
# DESCRIPTION: program that will inplement stochastic gradient descent algorithm
# for a one layer neural network
# VERSION: 0.0.1v
#
###############################################################################

def n_net_one_split(X_mat, y_vec, max_epochs, n_hidden_units, is_subtrain):
    # divide X_mat and y_vec into train, validation and test

    # initilize V_mat (weight matrix n_features x n_hidden_units)
        # used to predict hidden units given input

    # initilize w_vec (weight vector n_hidden_units size)
        # used to predict output given hidden units

    # loop over epochs (k=1 to max_epochs)
        # update the parameters using the gradients with respect to each
        # subtrain observation

        # loop over ata points in subtrain data set

            # compute the gradients of V-mat/w-vec with respect to a a
            # single observation in the subtrain set

            # update V.mat/w.vec by taking a step (scaled by step.size)
            # in the negative gradient direction

        # compute the logistic loss on the subtrain/validation sets
        # store value in loss_values (log loss formula log[1+exp(-y_tild * real_pred)])

    # return outputs (loss_values, V_mat, w_vec)