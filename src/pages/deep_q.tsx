import type { NextPage } from "next";
import Head from "next/head";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';

const DeepQ: NextPage = () => {

    const DeepModel = `
##---------------Sources-------------------------##
# DeepQ Learning with PyTorch: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# DeepQ for OpenAI Atari environments https://github.com/deepanshut041/Reinforcement-Learning 
##------------------------------------------------##

import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # TODO DM Changed kernel and stride to better fit our standard 28 x 40 size. 
        # Currently setup to take Atari 84x84 images
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)`

    const DeepAgent = `
##---------------Sources-------------------------##
# DeepQ Learning with PyTorch: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# DeepQ Image Processing for GymRetro:  https://github.com/deepanshut041/Reinforcement-Learning 
##------------------------------------------------##


import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
sys.path.append(os.path.abspath('..'))
script_dir = os.path.dirname(__file__)
sys.path.append(script_dir + '/../agents')
sys.path.append(script_dir + '/../interface')
sys.path.append(script_dir + '/../models')

from learning.deep_q_buffer import ReplayBuffer
from learning import train_deep_q
from agents.agent_base import * 
from interface.checkpoint import *
from models.deep_q_model import DQN
from learning.train_deep_q import stack_frames
from datetime import datetime, date
from interface.action_space import *



# DeepQ Neural Network. 
class DeepQ(AgentBase):

    # TODO Empty Constructor
    def __init__(self, model=None):
        """Initialize an Agent object.
        
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr (float): learning rate 
            update_every (int): how often to update the network
            replay_after (int): After which replay to be started
            model(Model): Pytorch Model
        """
        input_shape = (4, 84, 84) #stacked frames x w x h
        self.action_size = 7 #len(possible_actions)
        self.seed = 0 #random.seed(seed)
        self.buffer_size = 100000
        self.batch_size = 32
        self.gamma = 0.99
        self.lr = 0.0001
        self.update_every = 100
        self.replay_after = 10000
        if model is None:
            self.DQN = DQN
        else:
            self.DQN=torch.load(model)
        self.tau = 1e-3
        self.device = 'cpu'
        
        # Q-Network
        self.policy_net = self.DQN(input_shape, self.action_size).to(self.device)
        self.target_net = self.DQN(input_shape, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0]
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)
        something = 0

    def load(self, filename):
        None



    def save(self, epoch):
        date = str(datetime.now().date()) 
        torch.save(self.DQN, f'results/checkpoints/training/{self.name()}_{date}_e{str(epoch)}' )

    def train(self, env, n_episodes, reward, render, ckpt, save_rate):
        train_deep_q.train(self, env, n_episodes, reward, render, ckpt, save_rate)

    def decide(self, ob, info, counter) -> list:
        None
    

    # Returns name of agent as a string
    def name(self) -> str:
        return "DeepQ"`

    const DeepTraining = `
##---------------Sources-------------------------##
# DeepQ Learning with PyTorch: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# DeepQ Image Processing for GymRetro:  https://github.com/deepanshut041/Reinforcement-Learning 
# Helper Functions for Gym Retro: https://github.com/moversti/sonicNEAT 
##------------------------------------------------##

import time
import retro
import random
import torch
import numpy as np
from collections import deque
import math
import os

import sys
script_dir = os.path.dirname(__file__)
sys.path.append('../../')
sys.path.append(script_dir + '/../agents')
sys.path.append(script_dir + '/../interface')


from all_agents import * 
from deep_q_agent import *
from models.deep_q_model import DQN
from vision.image_processing import preprocess_frame, stack_frame
from interface.action_space import *
from vision.greyImageViewer import GreyImageViewer
from vision.controllerViewer import ControllerViewer



def stack_frames(frames, state, is_new=False):
    """Stacks frames for broader input of environment."""

    frame = preprocess_frame(state)
    frames = stack_frame(frames, frame, is_new)
    return frames


def train(agent, env, n_episodes=1000, reward='contest', render=False, ckpt=None, save_rate=10):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    possible_actions = ActionSpace().BUTTONS


    UPDATE_TARGET = 10000  # After which thershold replay to be started 
    EPS_START = 0.99       # starting value of epsilon
    EPS_END = 0.01         # Ending value of epsilon
    EPS_DECAY = 100         # Rate by which epsilon to be decayed


    # Initialize Agent
    start_epoch = 0
    scores = []
    scores_window = deque(maxlen=20)

    epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

    for i_episode in range(start_epoch + 1, n_episodes+1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)

        # Punish the agent for not moving forward
        prev_state = {}
        steps_stuck = 0
        timestamp = 0

        while timestamp < 10000:
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            score += reward

            timestamp += 1

            # Punish the agent for standing still for too long.
            if (prev_state == info):
                steps_stuck += 1
            else:
                steps_stuck = 0
            prev_state = info
    
            if (steps_stuck > 20):
                reward -= 1
            
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print (i_episode)
        if (i_episode % save_rate == 0):
            agent.save(i_episode)
    return scores


# Untrained Agent
# env.viewer = None
# state = stack_frames(None, env.reset(), True) 
# for j in range(10000):
#     env.render(close=False)
#     action = agent.act(state, eps=0.91)
#     next_state, reward, done, _ = env.step(possible_actions[action])
#     state = stack_frames(state, next_state, False)
#     if done:
#         env.reset()
#         break 
# env.render(close=True)

# Trained Agent
# train(50)
# print (scores)

# env.viewer = None
# state = stack_frames(None, env.reset(), True) 
# for j in range(10000):
#     env.render(close=False)
#     action = agent.act(state, eps=0.1)
#     next_state, reward, done, _ = env.step(possible_actions[action])
#     state = stack_frames(state, next_state, False)
#     if done:
#         env.reset()
#         break 
# env.render(close=True)`

  return (
    <>
      <Head>
        <title>DeepQ Implementation</title>
        <meta name="description" content="A page detailing the DeepQ implementation with code and demo." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">DeepQ Implemenation</h1>
            <p className="text-center mb-4">The third step in our project was to create a DeepQ implementation. DeepQ is a form of reinforcement learning. A reinforcement learning task focuses on training agents to interact within an environment. The agent arrives at different scenarios known as states by performing actions. Actions lead to rewards, which could be positive or negative. Let’s say we know the expected reward of each action at every step. This would essentially be like a cheat sheet for the agent! Our agent will know exactly which action to perform. It will perform the sequence of actions that will eventually generate the maximum total reward. This total reward is also called the Q-value. The Q-value strategy is calculated by a complex equation known as the Bellman Equation, which we will leave out for simplicity. Essentially, you try to maximize your reward by calculating rewards from all the possible states at the next time step. If you do this iteratively, you have Q-Learning!</p>
            <p className="text-center mb-4">Deep Q takes this a step further by using a neural network to calculate these action-reward pairs for each input state in parallel. It is typically several convolutional layers to process input images, followed by several fully connected layers to map estimated Q values to all possible actions. The network chooses the max Q value to decide the agents next action. Following the action, it receives a ground truth Q value. Through backpropagation, we minimize the loss between the estimated Q and the ground truth Q value. This is training! Eventually, our agent will learn the appropriate action to take relative to its current state, resulting in the greatest reward!</p>
            <p className="text-center flex-wrap">Source: <a target="_blank" href="https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/" rel="noopener noreferrer" className="text-yellow-400 underline hover:text-yellow-600">Analytic Vidhya</a></p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
                <video autoPlay muted loop className="max-h-96">         
                    <source src="/deepQ.mp4" type="video/mp4"/>       
                </video>
            </div>
        </article>
        <article>
          <h2 className="text-center text-yellow-400 p-6 md:px-8 lg:px-16">Our Code:</h2>
          <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6">DeepQ Model Class:</h3>
            <SyntaxHighlighter
              showLineNumbers
              style={vscDarkPlus}
              languag="python">
                {DeepModel}
            </SyntaxHighlighter>
            </div>
            </article>
            <article>
            <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6">DeepQ Agent Class:</h3>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {DeepAgent} 
            </SyntaxHighlighter>
            </div>
            </article>
            <article>
            <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6">DeepQ Traing Class:</h3>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {DeepTraining} 
            </SyntaxHighlighter>
          </div>
        </article>
      </main>
    </>
  );
};

export default DeepQ;