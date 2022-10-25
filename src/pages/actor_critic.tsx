import type { NextPage } from "next";
import Head from "next/head";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';


const ActorCritic: NextPage = () => {

  const ActorCriticAgent = `# based off of these pages:
# MNIST CNN:	https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
# MNIST CNN:	https://github.com/stabgan/CNN-classification-of-MNIST-dataset-using-pyTorch/blob/master/cnn.py
# Actor Critic: https://github.com/nikhilbarhate99/Actor-Critic-PyTorch/blob/master/model.py

#from importlib.metadata import distribution, requires
import sys
import os
import torch 
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.append(os.path.abspath('../agents'))
sys.path.append(os.path.abspath('../interface'))
sys.path.append(os.path.abspath('../learning'))

from agent_base import *
from action_space import *
from train_actor_critic import * 

# A decision making agent implemented as a neural network based on the Actor Critic Model.
# Trained using the Actor Critic Method.
class ActorCritic(AgentBase):

    # ----------------------- Nested Classes ----------------------------------

    class Actor(torch.nn.Module):
        def __init__(self):
            self.make_model()
        
        def make_model(self):
            super(ActorCritic.Actor, self).__init__()

            # --- Skip Input Layer ---
            # TODO: define input layer explicitly
            # For now, input layer will be created on the 1st forward pass.

            # *** Shouldn't we use Conv3d for colored images? ***
            # *** No Conv2d is for both gray and colored images ***

            # --- Hidden Layer 1 ---
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=3,		# 3 channel color image
                    out_channels=16,	# I think this means each convolutional kernel will output 16 nodes. See below ***
                    kernel_size=5,		# Size of each kernel in both dimentions: 5x5. I think this should be the size of each object in the game.
                    stride=1,			# Create a kernel on every so many pixels. 1 means don't skip any pixels.
                    padding=2,			# ???
                ),
                torch.nn.ReLU(),		# activation function. experiment with this. Try sigmoid.
                torch.nn.MaxPool2d(2),	# Pick the highest outputs of each group of 2x2 nodes.
            )

            # --- Hidden Layer 2 ---
            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=16,		# TODO: This should equal to out_channels from the previous layer. Should not be hard coded.		
                    out_channels=32,	# TODO: experiment with this
                    kernel_size=5,		# TODO: experiment with this
                    stride=1,			# 
                    padding=2,			# 
                ),
                torch.nn.ReLU(),		# 
                torch.nn.MaxPool2d(2),	# 
                )

            # --- Output Layers ---
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=143360,
                    out_features=ActionSpace.get_n_moves(),	# 1 output for each possible move. See class ActionSpace
                ),
                torch.nn.Sigmoid(),							# Maybe use sigmoid to cap range to [0.0, 1.0]
            )

            # *** I think out_channels is the number of output nodes that each kernel outputs.
            # For example, out_channels=16 creates 16 output nodes for each kernel. 
            # In any NN, each node extracts a feature from its input.
            # out_channels says how many different features to extract from that kernel.
            # This is like saying 16 different features will be extracted from each kernel.
            # The more features, the more the NN can learn.
            # Should this number correspond to the the number of output classes???
            # Its worth testing.

            return

        def forward(self, x):
            x = x.reshape((3, 224, 320))
            x = torch.unsqueeze(x, 0)
            x = x.float()
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            x = self.actor(x)		# Get action from actor layer
            distribution = Categorical(F.softmax(x, dim=1))	# TODO: Try dim=-1
            return distribution
        
    class Critic(torch.nn.Module):
        def __init__(self):
            self.make_model()
        
        def make_model(self):
            super(ActorCritic.Critic, self).__init__()

            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=3,		# 3 channel color image
                    out_channels=16,	# I think this means each convolutional kernel will output 16 nodes. See below ***
                    kernel_size=5,		# Size of each kernel in both dimentions: 5x5. I think this should be the size of each object in the game.
                    stride=1,			# Create a kernel on every so many pixels. 1 means don't skip any pixels.
                    padding=2,			# ???
                ),
                torch.nn.ReLU(),		# activation function. experiment with this. Try sigmoid.
                torch.nn.MaxPool2d(2),	# Pick the highest outputs of each group of 2x2 nodes.
            )

            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=16,		# TODO: This should equal to out_channels from the previous layer. Should not be hard coded.		
                    out_channels=32,	# TODO: experiment with this
                    kernel_size=5,		# TODO: experiment with this
                    stride=1,			# 
                    padding=2,			# 
                ),
                torch.nn.ReLU(),		# 
                torch.nn.MaxPool2d(2),	# 
                )

            self.critic = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=143360,
                    out_features=1,	# 1 output for each possible move. See class ActionSpace
                ),
                torch.nn.Sigmoid(),							# Maybe use sigmoid to cap range to [0.0, 1.0]
            )

        def forward(self, x):
            x = x.reshape((3, 224, 320))
            x = torch.unsqueeze(x, 0)
            x = x.float()
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            x = self.critic(x)		# Get action from actor layer

            return x
    
    # --------------------------------- METHODS -------------------------------

    # Creates an untrained neural network.
    # Weights and biases are initialized to random numbers.
    # image_size is a tuple in the form (3, width, height).
    # Inputs RGB images.
    # This initializes input layer of NN to match the input image.
    def __init__(self):
        self.make_model()
        
    # Creates a neural network and initializes it with random weights and biases.
    def make_model(self):
        self.actor = ActorCritic.Actor()
        self.critic = ActorCritic.Critic()

        self.actor.make_model()
        self.critic.make_model()

    # Executes a forward pass on input tensor x.
    # 1st dimension of x should be the batches of samples. 
    # The size of the 1st dimention is the batch size (1 in our case).
    # 2nd, 3rd and 4th dimensions are the dimensions of the image. This is implicitly determined on the 1st forward pass.
    # 
    # 1st - batches (1 sample per batch)
    # 2nd - number of color channels (3 for color, 1 for gray)
    # 3rd - image height in pixels
    # 4th - image width in pixels
    # 
    # *** This method must be implemented so that torch.nn.Module can execute forward passes. ***
    # It must be spelled "forward"
    # It must accept 1 parameter which is a torch.Tensor
    def forward(self, x):
        output = self.actor.forward(x)
        
        return output

    # Loads a NN from a file
    def load(self, filename) -> None:
        self.actor = torch.load('actor_19000.pt')
        self.critic = torch.load('critic_19000.pt')
    
    # Saves NN to a file
    # filename - file to save model in. Extension should be .pth 
    def save(self, filename) -> None:
        torch.save(self.state_dict(), filename)
    
    # Trains NN for one epoch
    def train(self, env, n_epochs, reward, render, ckpt, save_rate) -> None:
        ActorCriticTrainer.train_actor_critic(agent=self, env=env, n_epochs=n_epochs, reward=reward, render=render, checkpoint=ckpt, save_rate=save_rate)
    
    # Chooses an action based on the current state of the game.
    # obs - numpy.ndarray representing the game frame. 
    # info - information about the game state like time, position, and ring count.
    # returns - array of button presses. See class ActionSpace.
    def decide(self, obs, info) -> list:
        
        # 1.) Compute forward pass
        # TODO: Compute on GPU if available. See if that is faster.
        # output = output.cuda()
        img = torch.from_numpy(obs)

        output = self.forward(img)
        
        output = output.sample()

        # 2.) Convert int to array of button presses.
        buttons = ActionSpace.move(output.cpu().item())
        
        return buttons
        
    # Returns name of agent as a string
    def name(self) -> str:
            return "ActorCritic"
    
    def to_string(self) -> str:
        return self.name()

    # Moves data from current memory to 'device' memory
    # ex: agent.move_to(torch.KGPU) will move neural network data to GPU memory. 
    # If sucessfull, all operations on this NN will be executed on that device (CPU or GPU).
    # Internal fields will be moved. The object itself does not need to be reassigned like tensors do.
    def to(self, device) -> None:
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)`

    const ActorCriticTrain = `# Actor Critic using Cart Pole: https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py

import sys 
import os
import retro
import time
import torch 
import torch.optim as optim
import cv2 as cv

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(script_dir + '/../../data')

sys.path.append(script_dir + '/../interface')

from action_space import *
from reward_system import * 

class ActorCriticTrainer:
    def compute_returns(next_value, rewards, masks, gamma=0.99):
        R = next_value
        size = rewards.size()[0]
        returns = torch.zeros(size, device=rewards.device)
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns[step]

        return returns
    
    def train_actor_critic(agent, env, n_epochs, reward, render, checkpoint, save_rate):
        print("=== Training Actor Critic ===")
        
        optimizerA = optim.Adam(agent.actor.parameters(), lr = 0.02)
        optimizerC = optim.Adam(agent.critic.parameters(), lr = 0.02)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent.to(device)

        state = env.reset()

        reward_system = RewardSystem()
        
        # In each iteration, the nn is optimized once.
        for epoch in range(n_epochs):
            print("--- epoch =", epoch, "---")

            if epoch % 50 == 0:
                state = env.reset()
                
            # If n_steps is too large, it can cause memory errors
            n_steps = 200
            log_probs = torch.zeros(n_steps, device=device)
            values = torch.zeros(n_steps, device=device)
            rewards = torch.zeros(n_steps, device=device)
            masks = torch.zeros(n_steps, device=device)
            entropy = 0

            running_reward = 0
            next_state, reward, done, info = env.step(ActionSpace.stand_still())	# make a passive move to initialize values

            # --- Initialize Reward System with current game ---
            reward_system.init(info)
            reward = reward_system.calc_reward(next_state, info, ActionSpace.stand_still())

            jump_counter = 0

            for i in range(n_steps):
                if render:
                    env.render()

                # --- Forward Pass ---
                state = cv.resize(state, [int(244/4), int(320/4)])
                cv.imshow("img", state)
                cv.waitKey(1)
                state = torch.FloatTensor(state).to(device)
                dist = agent.actor(state)
                value = agent.critic(state)

                action = dist.sample()
                buttons = ActionSpace.move(action)
                next_state, reward, done, info = env.step(buttons)
                reward = reward_system.calc_reward(next_state, info, buttons)
                
                if ActionSpace.is_jump(buttons):
                    jump_counter += 1

                if done:
                    state = env.reset()
                    next_state, reward, done, info = env.step(ActionSpace.stand_still())	# make a passive move to initialize values
                    reward = reward_system.calc_reward(next_state, info, ActionSpace.stand_still())

                # --- Calc Reward ---
                running_reward += reward
                log_prob = dist.log_prob(action).unsqueeze(0)
                entropy += dist.entropy().mean()

                # --- Accumulate Data for Loss ---
                log_probs[i] = log_prob
                values[i] = value
                rewards[i] = reward
                masks[i] = 1-done

                state = next_state

                if done:
                    print('Epoch: {}, Score: {}'.format(epoch, i))
                    break

            print("\t\average reward per step:", running_reward / n_steps)
            print("\t\tjump count =", jump_counter)

            next_state = cv.resize(next_state, [int(244/4), int(320/4)])
            next_state = torch.FloatTensor(next_state).to(device)
            next_value = agent.critic(next_state)
            returns = ActorCriticTrainer.compute_returns(next_value, rewards, masks)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            
            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()

            if epoch % 1000 == 0:
                torch.save(agent.actor, 'actor_' + str(epoch) + '.pt')
                torch.save(agent.critic, 'critic_' + str(epoch) + '.pt')

        torch.save(agent.actor, 'actor.pt')
        torch.save(agent.critic, 'critic.pt')

        env.close()`


  return (
    <>
      <Head>
        <title>Actor-Critic Implementation</title>
        <meta name="description" content="A page detailing the Actor-Critic implementation with code and demo." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Actor-Critic Implemenation</h1>
            <p className="text-center">The next step in our project was to create an Actor-Critic implementation. The Actor-Critic model is a form of reinforcement learning that seperates the value calculation from the policy decisions. The Actor, in this model, determines which policies to make and makes decisions based on those policies. The Critic then determines the estimated effects of the decisions made and provides feedback to the Actor through temporal difference errors.</p>
          </div>
        </article>
        <article>
          <h2 className="text-center text-yellow-400 p-6 md:px-8 lg:px-16">Our Code:</h2>
            <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6">Actor-Critic Agent Class:</h3>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {ActorCriticAgent} 
            </SyntaxHighlighter>
            </div>
            </article>
            <article>
            <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6">Actor-Critic Traing Class:</h3>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {ActorCriticTrain} 
            </SyntaxHighlighter>
          </div>
        </article>
      </main>
    </>
  );
};

export default ActorCritic;