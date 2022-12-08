import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";

import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';


const Home: NextPage = () => {
  return (
    <>
      <Head>
        <title>SuperSonicAI</title>
        <meta name="description" content="Welcome to the Super Sonic AI project." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">SuperSonicAI: Deep Reinforcement Learning for Low Dimensional Environments</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Project Overview</h2>
            <p className="text-center">This project is an application that plays Sonic the Hedgehog Genesis (Sonic)  using artificial intelligence. It is able to generalize to any other game with similar game mechanics and controls. The application executes a decision making algorithm which plays the game by training neural networks to see what the user sees, extract useful information and make decisions to navigate the environment. Our application interfaces with the game using Gym Retro, an open source platform for reinforcement learning training and visualization. The platform solves two very challenging problems: The first is that the game was compiled to run on a processor which is incompatible with modern computers. The second is that its interface was developed to work directly with monitors and game controllers, and not with other programs. This platform allows a program to interface with supported games. Gym Retro runs the game in a builtin emulator as a child process. All this is conveniently encapsulated in an object oriented API made available by the Gym Retro package.</p>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Approach</h2>
            <p className="text-center">In order to develop the optimal agent to play Sonic, the team constructed a stable software architecture as a platform for experimentation across implementations of reinforcement learning agents, computer vision image processing techniques, and helper functions to support explainability. This experimentation led us to a solution that can finish the first level of Sonic and generalizes fairly well to unseen environments. This solution involves generating a synthetic dataset of images from the game to train a DeepLab V3 semantic segmentation model. We then apply this trained model to the Sonic emulator as a preprocessing step to feed segmented images into a Deep Q Learning Agent. The Deep Q Learning Agent was then on several levels of Sonic to develop the optimal policy of state-action pairs to support generalization on unseen environments. This final approach is open-source and can be replicated by running the following command line arguments below from the root project directory.</p>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Implementation</h2>
            <p className="text-center">The following utilizes the default command line arguments as input, except where specified. Defaults can be found in the driver code files below, or by adding the -h command after the command to view defaults and options for each argument</p>
            <h3 className="text-yellow-400 text-center mt-4">Install Software Architecture</h3>
            <SyntaxHighlighter
              style={vscDarkPlus}
              language="bash">
                git clone https://git.cs.vt.edu/dmath010/supersonicai.git
            </SyntaxHighlighter>
            <SyntaxHighlighter
              style={vscDarkPlus}
              language="bash">
                pip install -r requirements.txt
            </SyntaxHighlighter>
            <h3 className="text-yellow-400 text-center mt-4">Generate Synthetic Image Dataset</h3>
            <SyntaxHighlighter
              style={vscDarkPlus}
              language="bash">
                python source/drivers/generate_dataset.py -n 20000
            </SyntaxHighlighter>
            <h3 className="text-yellow-400 text-center mt-4">Train Semantic Segmentation Model</h3>
            <SyntaxHighlighter
              style={vscDarkPlus}
              language="bash">
                python source/drivers/deeplab_train.py -e 45
            </SyntaxHighlighter>
            <h3 className="text-yellow-400 text-center mt-4">Train Deep Q Learning Agent</h3>
            <SyntaxHighlighter
              style={vscDarkPlus}
              language="bash">
                python source/drivers/train.py --agent_type dqn --level 1 --epochs 100  --seg results/deeplab_ckpts/SegmentationModel.pt
            </SyntaxHighlighter>
            <h3 className="text-yellow-400 text-center mt-4">Watch your Agent Play</h3>
            <SyntaxHighlighter
              style={vscDarkPlus}
              language="bash">
                python source/drivers/play.py --seg results/deeplab_ckpts/SegmentationModel.pt --render --level 1
            </SyntaxHighlighter>
          </div>
        </article>
        <article className="container mx-auto flex flex-col items-center p-4 md:px-8 lg:px-16">
          <h2 className="text-yellow-400  text-center m-1">Team</h2>
            <div className="flex flex-col md:flex-row text-center md:gap-8">
              <div className="flex flex-col">
                <p><b className="text-yellow-400">Abdulmaged Ba Gubair</b> - Software Engineer, Systems</p>
                <p><b className="text-yellow-400">Blake Barnhill</b> - Front End Integration/Web Development</p>
                <p><b className="text-yellow-400">Kevin Chahine</b> - Software Engineer, Reinforcement Learning</p>
              </div>
              <div className="flex flex-col">
                <p><b className="text-yellow-400">David Cho</b> - Software Engineer, Reinforcement Learning</p>
                <p><b className="text-yellow-400">Danny Mathieson</b> - Software Engineer, Computer Vision</p>
                <p><b className="text-yellow-400">Drew Klaubert</b> - Software Engineer, Machine Learning</p>
              </div>
            </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/initialization"}>Phase 1: Initialization -&gt;</Link>
          </button>
        </article>
      </main>
    </>
  );
};

export default Home;
