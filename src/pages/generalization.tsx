import type { NextPage } from "next";
import Head from "next/head";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';
import Link from "next/link";

const DeepQ: NextPage = () => {

    const DeepModel = `##---------------Sources-------------------------##
# DeepQ Learning with PyTorch: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# DeepQ for OpenAI Atari environments https://github.com/deepanshut041/Reinforcement-Learning 
##------------------------------------------------##

import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(script_dir + "/../..")

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, seed=0):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.seed = seed
        
        
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

    const DeepAgent = `##---------------Sources-------------------------##
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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(script_dir + "/../..")

sys.path.append(os.path.abspath(project_dir + '/source/agents'))
sys.path.append(os.path.abspath(project_dir + '/source/interface'))
sys.path.append(os.path.abspath(project_dir + '/source/learning'))
sys.path.append(os.path.abspath(project_dir + '/source/models'))

from agent_base import * 
from deep_q_buffer import *
from train_deep_q import *
from deep_q_model import DQN
from datetime import datetime, date
from action_space import *

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
        input_shape = (4, 84, 84) #stacked frames x channels x w x h
        self.action_size = 7 #len(possible_actions)
        self.seed = 0 #random.seed(seed)
        self.buffer_size = 100000
        self.batch_size = 32
        self.gamma = 0.99
        self.lr = 0.00001
        self.update_every = 100
        self.replay_after = 10000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.DQN = DQN
        self.tau = 1e-3

        
        # Q-Network
    
        self.policy_net = self.DQN(input_shape, self.action_size, self.seed).to(self.device)
        if model is not None:
            os.chdir(script_dir)
            os.chdir('..')
            os.chdir('..')
            root = os.getcwd()
            checkpoint = torch.load(model, map_location=self.device)

            self.policy_net.load_state_dict(checkpoint['model_state_dict'])

            #self.policy_net.load_state_dict(torch.load(os.path.join(root, model), map_location=self.device), strict=False)
        self.target_net = self.DQN(input_shape, self.action_size, self.seed).to(self.device)
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
                
    def act(self, state, eps=0.03):
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

    def save(self, filename, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.policy_net.state_dict(),
            }, filename)
        
    def train(self, env, n_episodes, reward_system, render, ckpt, save_rate):
        DeepQTrainer.train(self, env, n_episodes, reward_system, render, ckpt, save_rate)
    
    def decide(self, ob, info) -> list:
        # Quick Fix
        if hasattr(self, '__prev_state'):
            self.__prev_state = DeepQTrainer.stack_frames(self.__prev_state, ob, False)
        else:
            self.__prev_state = DeepQTrainer.stack_frames(None, ob, True) 
        
        move = self.act(self.__prev_state)

        return ActionSpace.move(move)

    # Returns name of agent as a string
    def name(self) -> str:
        return "DeepQ"

    # Moves data from current memory to 'device' memory
    # ex: agent.move_to(torch.cuda()) will move neural network data to GPU memory. 
    # If sucessfull, all operations on this NN will be executed on that device (CPU or GPU).
    # Internal fields will be moved. The object itself does not need to be reassigned like tensors do.
    def to(self, device) -> None:
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)`

    const DeepTraining = `##---------------Sources-------------------------##
# DeepQ Learning with PyTorch: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# DeepQ Image Processing for GymRetro:  https://github.com/deepanshut041/Reinforcement-Learning 
# Helper Functions for Gym Retro: https://github.com/moversti/sonicNEAT 
##------------------------------------------------##

from fileinput import filename
import time
import retro
import random
import torch
import numpy as np
from collections import deque
import math
import os
import sys
from datetime import datetime, date

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(script_dir + "/../..")

sys.path.append(os.path.abspath(project_dir + '/source/agents'))
sys.path.append(os.path.abspath(project_dir + '/source/interface'))
sys.path.append(os.path.abspath(project_dir + '/source/learning'))
sys.path.append(os.path.abspath(project_dir + '/source/models'))
sys.path.append(os.path.abspath(project_dir + '/source/vision'))

from all_agents import * 
from checkpoint import *
from deep_q_agent import *
from deep_q_model import DQN
from image_processing import preprocess_frame, stack_frame
from action_space import *
from greyImageViewer import GreyImageViewer
from controllerViewer import ControllerViewer
from reward_system import *

class DeepQTrainer:
    def stack_frames(frames, state, is_new=False):
        """Stacks frames for broader input of environment."""

        frame = preprocess_frame(state)
        frames = stack_frame(frames, frame, is_new)
        return frames


    def train(agent, env, n_episodes=1000, reward_system=RewardSystem.Contest, render=False, ckpt=None, save_rate=10):
        """
        Params
        ======
            n_episodes (int): maximum number of training episodes
        """
        # if gpu is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", device)

        UPDATE_TARGET = 10000	# After which thershold replay to be started 
        EPS_START = 0.99		# starting value of epsilon
        EPS_END = 0.01			# Ending value of epsilon
        EPS_DECAY = 100			# Rate by which epsilon to be decayed

        # Initialize Agent
        
        start_epoch = 0
        scores = []
        best_ckpt_score = -9999999	# initialize to a very small number
        scores_window = deque(maxlen=20)

        # Initialize checkpoint
        ckpt = Checkpoint(agent)
        ckpt.make_dir() # Makes new directory if it does not exist

        epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)
        for i_episode in range(start_epoch + 1, n_episodes+1):
            state = DeepQTrainer.stack_frames(None, env.reset(), True)
            next_state, reward, done, info = env.step(ActionSpace.stand_still())	# make a passive move to initialize data

            score = 0
            eps = epsilon_by_epsiode(i_episode)
            reward_system.init(info)

            # Punish the agent for not moving forward
            prev_state = {}
            steps_stuck = 0
            timestamp = 0

            while timestamp < 5000:
                action = agent.act(state, eps)
                next_state, reward, done, info = env.step(ActionSpace.move(action))
                reward = reward_system.calc_reward(info, ActionSpace.move(action))

                if render is True:
                    env.render()

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

                next_state = DeepQTrainer.stack_frames(state, next_state, False)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            
            scores_window.append(score)		# save most recent score
            scores.append(score)			# save most recent score
            print ("epoch:", i_episode, "score:", score)
        
            if (i_episode % save_rate == 0 and score > best_ckpt_score):
                ckpt.epoch = i_episode
                ckpt.score = score
                best_ckpt_score = score
                fn = ckpt.generate_path()
                agent.save(fn, i_episode)
                print(f"Saving checkpoint with new best score {best_ckpt_score}")
            
        return scores`

  return (
    <>
      <Head>
        <title>DeepQ Generalization</title>
        <meta name="description" content="What is DeepQ generalization with our full code and outcome." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">DeepQ Generalization</h1>
            <p className="text-center mb-4">The fourth step in our project was to create a DeepQ generalization. Generalization is a further expansion on our <Link href="/deep_q" className="text-yellow-400 underline hover:text-yellow-600">DeepQ Implementation.</Link></p>
            <p className="text-center mb-4">Generalization refers to a model&apos;s ability to adapt to novel data not seen during training. In reinforcement learning, this occurs when an agent utilizes a policy developed outside of the deployment environment. For our case, Sonic is trained on several levels, and the agent attempt to use a policy to conquer an unseen level. While this can decrease overall performance across environments, it is vital to developing viable agents for use in real world applications.</p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
                <h1 className="text-yellow-400 text-2xl m-2">Generalized Runs</h1>
                <p className="text-center mb-4">This is Sonic&apos;s first attempt at Level 2 after generalization training:</p>
                <video autoPlay muted loop className="max-h-96">         
                    <source src="/generalization_level2.mp4" type="video/mp4"/>       
                </video>
                <p className="text-center mb-4">As you can see, Sonic has a lot of problems when faced with a loop and has not figured out how to dodge enemies he hasn&apos;t seen before.</p>

                <p className="text-center mb-4">This is Sonic&apos;s first attempt at Level 4 after generalization training:</p>
                <video autoPlay muted loop className="max-h-96">         
                    <source src="/generalization_level4.mp4" type="video/mp4"/>       
                </video>
                <p className="text-center mb-4">As you can see, Sonic has a lot of problems with both the lava and the moving platforms, both of which he has not seen before.</p>
            </div>
        </article>
        <article>
          <h2 className="text-center text-yellow-400 p-6 md:px-8 lg:px-16">Our Code:</h2>
          <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6">DeepQ Model Class:</h3>
            <SyntaxHighlighter
              showLineNumbers
              style={vscDarkPlus}
              language="python">
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