import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";
import Image from "next/image";

import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';


const Home: NextPage = () => {

    const replayAgent = `# Various imports as needed with system paths set
import sys
import os
import retro

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(script_dir + "/../..")

sys.path.append(os.path.abspath(project_dir + '/source/agents'))
sys.path.append(os.path.abspath(project_dir + '/source/interface'))

from agent_base import *

class ReplayAgent(AgentBase):
    def __init__(self, filename='SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2'):
        # filename = level.game_name() + '-' + level.to_state() + '-0000.bk2'

        # We need to figure out how to get the proper stage from play.py
        stage = filename
        movie_path = f'{project_dir}/source/datasets/contest/' + stage
        self.movie = retro.Movie(movie_path)
        self.movie.step()
        self.moves = []
        while self.movie.step():
            keys = []
            for i in range(12):
                keys.append(self.movie.get_key(i, 0))
            self.moves.append(keys)
        self.i = -1

    def save(self, filename):
        None	# nothing to do here

    def train(self):
        None	# nothing to do here

    def decide(self, obs, info) -> list:
        self.i += 1
        return self.moves[self.i]

    # Returns name of agent as a string
    def name(self) -> str:
        return "ReplayAgent"

    def to_string(self) -> str:
        return self.name()`

  return (
    <>
      <Head>
        <title>Replay Agent</title>
        <meta name="description" content="Replay Agent is a class that supports development of reinforcement learning agent models by rendering a human-playable action sequence." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Replay Agent</h1>
            <p className="text-center">Replay Agent is a class that supports development of reinforcement learning agent models by rendering a human-playable action sequence. This agent simply follows a sequence of moves determined by the button presses of a human playing the game at an earlier date. This allows us to start training from various stages in the level to overcome obstacles that are particularly difficult for the agent. In practice, we run the Replay Agent for a determined number of timesteps to encounter the obstacle, then train our Reinforcement Learning agent from that obstacle. This tailors our training for the most difficult cases and leads to optimal model performance.</p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <video autoPlay muted loop className="max-h-96">         
                <source src="/replay_agent.mp4" type="video/mp4"/>       
            </video>
            </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/deep_q"}>DeepQ Agent -&gt;</Link>
          </button>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Replay Agent Code</h2>
            <p>source/agents/replay_agent.py</p>
            <SyntaxHighlighter
                showLineNumbers
                style={vscDarkPlus}
                language="python">
                {replayAgent}
            </SyntaxHighlighter>
          </div>
        </article>
        
      </main>
    </>
  );
};

export default Home;
