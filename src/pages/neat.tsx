import type { NextPage } from "next";
import Head from "next/head";
import Header from "../components/header";
// import { Prism } from '@mantine/prism';
// import { CopyBlock, dracula} from 'react-code-blocks';
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import darcula from 'react-syntax-highlighter/dist/cjs/styles/prism';


const Home: NextPage = () => {

    const neatCode = `
import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')

imgarray = []

xpos_end = 0


    def eval_genomes(genomes, config):

    # Iterate through all genomes (network configurations)
    # We set this number to 20 in the Neat config file
    for genome_id, genome in genomes:

        #Image variable (input to network)
        ob = env.reset()
        # 12 array of buttons (output of network)
        ac = env.action_space.sample()

        # Size of image w,h,c
        # For Sonic, it is 224x320x3
        inx, iny, inc = env.observation_space.shape

        # Scale down resolution to 1/8, remove color dimension
        # So our netork takes in a 28x40 matrix
        inx = int(inx/8)
        iny = int(iny/8)

        # This is the network for this current run
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            
        ##############
        # Metrics
        ##############
        # How successful our network is, defined by our reward function
        current_max_fitness = 0
        fitness_current = 0
        # Count frames
        frame = 0
        counter = 0
        # xpos is an easy way to check for completion of level
        xpos = 0
        xpos_max = 0
            
        done = False
        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        # Inner loop, runs
        while not done:
                
            #Start game
            env.render()
            frame += 1
            #scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            #scaledimg = cv2.resize(scaledimg, (iny, inx)) 


            # Resize our emulator frame to feed into network
            ob = cv2.resize(ob, (inx, iny))
            # Make grayscale (remove color dimension)
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)

            #Reshape for some reason, but necessary
            ob = np.reshape(ob, (inx,iny))
            #cv2.imshow('main', scaledimg)
            #cv2.waitKey(1) 

            # Puts the value of each pixel in a list, so our network
            # takes in a 1D array. Necessary for Neat
            imgarray = np.ndarray.flatten(ob)

            # This is train. Sends input (imgarray) and outputs button array
            nnOutput = net.activate(imgarray)
                
            # Increments emulator step by 1 with output of network (button array)
            ob, rew, done, info = env.step(nnOutput)

            #############
            # Completion Conditions (2) 1 is xpos based, the other is fitness based
            # We're using fitness based
            ###############

            # Records x position from info above
            #xpos = info['x']

            # This is the end of the level
            #xpos_end = info['screen_x_end']
                
            # Sonic gets a fitness point if hes been further right than ever before
            #if xpos > xpos_max:
                #fitness_current += 1
                #xpos_max = xpos
                
            # If he hits end, 10000 fitness points
            #if xpos == xpos_end and xpos > 500:
                #fitness_current += 100000
                #done = True
                
            # Fitness is the cumulative total of rewards
            fitness_current += rew
                
            # If he's still improving, keep going
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                
            # Done if he dies 3 times (default in scenario file)
            #  or if he doesn't imrpove over 250 iterations
            if done or counter == 250:
                done = True

                #print the genome and the fitness @ end of genome
                print(genome_id, fitness_current)
                    
            # Keep updating this genomes fitness to the current fitness until we're done
            genome.fitness = fitness_current
                    
                
# Config file modified from tutorial
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'config-feedforward')

# Set population from config file
p = neat.Population(config)


#Basic Stats
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
#Save checkpoint
p.add_reporter(neat.Checkpointer(10))

# Optimal network from NEAT run
winner = p.run(eval_genomes, 1)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)`


  return (
    <>
      <Head>
        <title>NEAT Implementation</title>
        <meta name="description" content="A page detailing the NEAT implementation with code and demo." />
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
          <h2 className="text-center text-yellow-400 p-6 md:px-8 lg:px-16">Our Code:</h2>
          <div className="m-6 md:mx-8 lg:mx-16">
          <SyntaxHighlighter 
            theme={darcula}
            language="python">
            {neatCode} 
            </SyntaxHighlighter>
          </div>
        </article>
      </main>
    </>
  );
};

export default Home;
