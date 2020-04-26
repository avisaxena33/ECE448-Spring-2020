# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, in_size, out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        The network should have the following architecture (in terms of hidden units):
        in_size -> 128 ->  out_size
        """
        super(NeuralNet, self).__init__()   # init module
        self.loss_fn = nn.CrossEntropyLoss()
        self.in_size = in_size
        self.out_size = out_size
        self.lrate = lrate
        self.hidden = nn.Linear(self.in_size, 128, True)   # inputs to hidden layer
        self.output = nn.Linear(128, self.out_size, True) # output layer
        self.sigmoid = nn.Sigmoid() # activation function
        self.optimizer = optim.SGD(self.get_parameters(), lr=self.lrate)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        # return self.net.parameters()
        return self.parameters()


    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = self.hidden(x)          # hidden linear combination -> sigmoid activation -> output linear combination -> sigmoid
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad() # clears the gradient buffer
        x = self(x)                # get ouput of network from given input
        loss = self.loss_fn(x, y)  # loss function and backwards
        loss.backward()  
        self.optimizer.step()      # optimizer step to update weights and all
        return loss

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, out_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """
    my_network = NeuralNet(1, len(train_set[0]), 3)    #initialize the net
    losses = []

    mean = train_set.mean()     # standardize the training and dev data based on training data stats
    std_dev = train_set.std()
    train_set = (train_set - mean) / std_dev
    dev_set = (dev_set - mean) / std_dev

    for i in range(n_iter+10):  # train network (n_iter+2 * batch_Size)
        loss_tensor = my_network.step(train_set, train_labels)
        loss = loss_tensor.item()
        losses.append(loss)

    output_tensor = my_network(dev_set)     # classify dev_set and find max indices
    indices = output_tensor.argmax(1)
    yhats = np.asarray(indices)

    return losses, yhats, my_network
