import gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim

import utils
from policies import QPolicy

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
        self.loss_fn = nn.MSELoss()
        self.in_size = statesize
        self.out_size = actionsize
        self.lrate = .01
        self.hidden = nn.Linear(self.in_size, 32, True)   # inputs to hidden layer
        self.hidden2 = nn.Linear(32, 32, True)
        self.output = nn.Linear(32, self.out_size, True) # output layer
        self.optimizer = optim.SGD(self.get_parameters(), lr=self.lrate)
        self.sigmoid = nn.Sigmoid() # activation function
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
        x = self.hidden(x)          # hidden linear combination -> sigmoid activation -> output linear combination -> sigmoid
        x = self.leakyrelu(x)
        x = self.hidden2(x)
        x = self.leakyrelu(x)
        x = self.output(x)
        return x

def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    my_model = NeuralNet(statesize, actionsize)
    return my_model

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
        curr_quality = self.qvals(state)
        next_state_max_quality = max(self.qvals(next_state))
        target = None
        if done and next_state[0] >= 0.5:
            reward = 1.0
            target = reward
        else:
            target = reward + self.gamma*next_state_max_quality
        updated_curr_quality = curr_quality + self.lr*(target - curr_quality)
        self.model.optimizer.zero_grad() # clears the gradient buffer
        loss = self.model.loss_fn(curr_quality, target)  # loss function and backwards
        loss.backward()  
        self.model.optimizer.step()      # optimizer step to update weights and all
        loss = loss.item()
        return loss

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
