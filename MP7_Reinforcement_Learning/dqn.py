import gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim

import utils
from policies import QPolicy

def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    return nn.Sequential(nn.Linear(statesize, 128, True), nn.ReLU(), nn.Linear(128, 128, True), nn.ReLU(), nn.Linear(128, actionsize, True))

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

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
        return np.asarray([qvals.numpy()])

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
        self.optimizer.zero_grad() # clears the gradient buffer
        curr_quality = self.model(torch.from_numpy(state).type(torch.FloatTensor))
        next_state_max_quality = max(self.model(torch.from_numpy(next_state).type(torch.FloatTensor)))
        target = None
        if done:
            reward = 1.0
            target = torch.from_numpy(np.asarray(reward)).type(torch.FloatTensor)
        else:
            target = reward + self.gamma*next_state_max_quality
        loss = self.loss_fn(curr_quality[action], target) # loss function and backwards
        loss.backward()  
        self.optimizer.step()      # optimizer step to update weights and all
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
