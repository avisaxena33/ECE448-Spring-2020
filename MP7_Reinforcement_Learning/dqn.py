import gym
import numpy as np
import torch
from torch import nn

import utils
from policies import QPolicy


def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    class NeuralNet(torch.nn.Module):
        def __init__(self, statesize, actionsize):
            """
            Initialize the layers of your neural network
            @param lrate: The learning rate for the model.
            @param loss_fn: The loss function
            @param in_size: Dimension of input
            @param out_size: Dimension of output
            """
            super(NeuralNet, self).__init__()
            self.loss_fn = nn.CrossEntropyLoss()
            self.in_size = statesize
            self.out_size = out_size
            self.lrate = lrate
            self.hidden = nn.Linear(self.in_size, 128, True)   # inputs to hidden layer
            self.hidden2 = nn.Linear(128, 128, True)
            self.output = nn.Linear(128, self.out_size, True) # output layer
            self.sigmoid = nn.Sigmoid() # activation function
            self.optimizer = optim.SGD(self.get_parameters(), lr=self.lrate, weight_decay=1e-10)
            self.tanh = nn.Tanh()
            self.relu = nn.ReLU()
            self.selu = nn.SELU()
            self.leakyrelu = nn.LeakyReLU()

        def get_parameters(self):
            """ Get the parameters of your network
            @return params: a list of tensors containing all parameters of the network
            """
            # return self.net.parameters()
            return self.parameters()

        def forward(self, x):
            """ A forward pass of your autoencoder
            @param x: an (N, in_size) torch tensor
            @return xhat: an (N, out_size) torch tensor of output from the network
            """
            #return torch.zeros(x.shape[0], 28*28)
            x = self.encoder(x)
            x = self.decoder(x)
            return x

        def step(self, x):
            # x [100, 784]
            """
            Performs one gradient step through a batch of data x with labels y
            @param x: an (N, in_size) torch tensor
            @return L: total empirical risk (mean of losses) at this time step as a float
            """
            #self.optimizer.zero_grad() # clears the gradient buffer
            #x = self(x)                # get ouput of network from given input
            #loss = self.loss_fn(x, y)  # loss function and backwards
            #loss.backward()  
            #self.optimizer.step()      # optimizer step to update weights and all
            #return loss


class DQNPolicy(QPolicy):
    """
    Function approximation via a deep network
    """

    def __init__(self, model, statesize, actionsize, lr, gamma):
        """
        Inititalize the dqn policy

        @param model: the nn.Module instance returned by make_dqn
        @param statesize: dimension of the input continuous state space.
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate 
        @param gamma: discount factor
        """
        super().__init__(statesize, actionsize, lr, gamma)
        self.model = model
        self.env = env

    def qvals(self, state):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.from_numpy(state).type(torch.FloatTensor)
            qvals = self.model(states)
        return qvals.numpy()

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """

    def save(self, outpath):
        """
        saves the model at the specified outpath
        """        
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    policy = DQNPolicy(make_dqn(statesize, actionsize), statesize, actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'models/dqn.model')
