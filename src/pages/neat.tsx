import type { NextPage } from "next";
import Head from "next/head";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';


const Home: NextPage = () => {

  const NeatFeedForward = `[NEAT]
fitness_criterion     = max
fitness_threshold     = 100000
pop_size              = 10
reset_on_extinction   = True

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.05
activation_options      = sigmoid gauss 
#abs clamped cube exp gauss hat identity inv log relu sigmoid sin softplus square tanh

# node aggregation options
aggregation_default     = random
aggregation_mutate_rate = 0.05
aggregation_options     = sum product min max mean median maxabs

# node bias options
bias_init_mean          = 0.05
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.1

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.2

feed_forward            = False
#initial_connection      = unconnected
initial_connection      = partial_nodirect 0.5

# node add/remove rates
node_add_prob           = 0.5
node_delete_prob        = 0.5

# network parameters
num_hidden              = 0
num_inputs              = 1120
num_outputs             = 12

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.05
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.1
response_mutate_rate    = 0.75
response_replace_rate   = 0.1

# connection weight options
weight_init_mean        = 0.1
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 2.5

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 50
species_elitism      = 0

[DefaultReproduction]
elitism            = 1
survival_threshold = 0.3`

    const NeatAgent = `import sys
import os

script_dir = os.path.dirname(__file__)

sys.path.append(script_dir + '/../agents')
sys.path.append(script_dir + '/../interface')
sys.path.append(script_dir + '/../learning')

from agent_base import *
from action_space import *
from train_neat import *

class NeatAgent(AgentBase):

  def load(self, filename):
    None
    # TODO:

  def save(self, filename):
    None
    # TODO:
    
  def train(self):
    train_neat()

  def decide(self, obs, info) -> list:
    buttons = ActionSpace.move_right()	

    return buttons

  # Returns name of agent as a string
  def name(self) -> str:
    return "NeatAgent"

  def to_string(self) -> str:
    return self.name()`
  
  const NeatTraining = `##---------------Sources-------------------------##
# Neat NN Implementation: https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT
# DeepQ Image Processing for GymRetro:  https://github.com/deepanshut041/Reinforcement-Learning 
# Helper Functions for Gym Retro: https://github.com/moversti/sonicNEAT 
##-----------------------------------------------##

import retro
import numpy as np
import cv2
import neat
import pickle
import sys
import os

from configparser import Interpolation
from inspect import getsourcefile

from vision.greyImageViewer import GreyImageViewer
from vision.controllerViewer import ControllerViewer

script_dir = os.path.dirname(__file__)
sys.path.append(script_dir + '/../agents')
sys.path.append(script_dir + '/../interface')

# Trains a NEAT NN.
def train_neat():
    env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1", scenario="contest", record='.')
    imgarray = []
    xpos_end = 0

    SEE_NETWORK_INPUT=True

    resume = True
    restore_file = "neat-checkpoint-32"

    viewer = GreyImageViewer()
    controllerViewer = ControllerViewer()

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            ob = env.reset()
            ac = env.action_space.sample()

            inx, iny, inc = env.observation_space.shape

            inx = int(inx / 8)
            iny = int(iny / 8)

            net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

            current_max_fitness = 0
            fitness_current = 0
            frame = 0
            counter = 0
            xpos = 0

            done = False

            while not done:

                env.render()
                frame += 1
                ob = cv2.resize(ob, (inx, iny))
                #ob = cv2.blur(ob, (3,3))
                ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)

                if SEE_NETWORK_INPUT:
                    img = ob.copy()
                    dst = (img.shape[0] * 8, img.shape[1] * 8)
                    img = cv2.resize(img, dst, interpolation=cv2.INTER_NEAREST)
                    img = np.flipud(img)
                    viewer.imshow(img)

                ob = np.reshape(ob, (inx, iny))

                imgarray = np.ndarray.flatten(ob)

                nnOutput = net.activate(imgarray)
                ac = env.action_to_array(nnOutput)
                # print(ac)
                controllerViewer.actionshow(ac)
                ob, rew, done, info = env.step(nnOutput)

                xpos = info['x']

                if xpos >= 60000:
                    fitness_current += 10000000
                    done = True

                fitness_current += rew

                if fitness_current > current_max_fitness:
                    current_max_fitness = fitness_current
                    counter = 0
                else:
                    counter += 1

                if done or counter == 250:
                    done = True
                    print(genome_id, fitness_current)

                genome.fitness = fitness_current

    # Get directory of current script. This directory is also the one contain 'config-feedforward'
    config_dir = os.path.dirname(getsourcefile(lambda:0))   

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      'source/models/neat-feedforward')
    if resume == True:
        p = neat.Checkpointer.restore_checkpoint(restore_file)
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1, filename_prefix='neat-checkpoint-'))

    winner = p.run(eval_genomes, 1)

    with open('winnerTEST.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

if __name__ == "__main__":
    train_neat()`


  return (
    <>
      <Head>
        <title>NEAT Implementation</title>
        <meta name="description" content="What is NEAT with our code and training outcome." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">NEAT Implemenation</h1>
            <p className="text-center">The first step in our project was to create a basic NEAT implementation. NEAT stands for NeuroEvolution of Augmenting Topologies, which is a form of evolutionary machine learning developed by researchers at MIT. The result is a fast but structured machine learning algorithm.</p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
                <video autoPlay muted loop className="max-h-96">         
                    <source src="/neat.mp4" type="video/mp4"/>       
                </video>
            </div>
        </article>
        <article>
          <h2 className="text-center text-yellow-400 p-6 md:px-8 lg:px-16">Our Code:</h2>
          <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6">NEAT Model Class:</h3>
            <SyntaxHighlighter
              showLineNumbers
              style={vscDarkPlus}
              languag="python">
                {NeatFeedForward}
            </SyntaxHighlighter>
            </div>
            </article>
            <article>
            <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6">NEAT Agent Class:</h3>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {NeatAgent} 
            </SyntaxHighlighter>
            </div>
            </article>
            <article>
            <div className="m-6 md:mx-8 lg:mx-16">
            <h3 className="text-yellow-400 pt-6">NEAT Traing Class:</h3>
            <SyntaxHighlighter 
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {NeatTraining} 
            </SyntaxHighlighter>
          </div>
        </article>
      </main>
    </>
  );
};

export default Home;