import type { NextPage } from "next";
import Head from "next/head";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';
import Image from "next/future/image";
import deepQVideo from "../../public/deepQ.mp4";


const DeepQ: NextPage = () => {

    const code = `# Implementation Coming Soon`

    

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
            <h1 className="text-yellow-400 text-2xl m-2">DeepQ Implemenation</h1>
            <p className="text-center mb-2">The third step in our project was to create a DeepQ implementation. The DeepQ model is A reinforcement learning task is about training agents to interact with an environment. The agent arrives at different scenarios known as states by performing actions. Actions lead to rewards which could be positive and negative. Let’s say we know the expected reward of each action at every step. This would essentially be like a cheat sheet for the agent! Our agent will know exactly which action to perform. It will perform the sequence of actions that will eventually generate the maximum total reward. This total reward is also called the Q-value. The Q value strategy is calculated by the complex Bellman Equation, which we will leave out for simplicity. Essentially, you try to maximize your reward by calculating rewards from all the possible states at the next time step. If you do this iteratively, you have Q-Learning!</p>
            <p className="text-center mb-2">Deep Q takes this a step further by using a neural network to calculate these action-reward pairs for each input state in parallel. It is typically several convolutional layers to process input images, followed by several fully connected layers to map estimated Q values to all possible actions. The network chooses the max Q value to decide the agents next action. Following the action, it receives a ground truth Q value. Through backpropagation, we minimize the loss between the estimated Q and the ground truth Q value. This is training! Eventually, our agent will learn the appropriate action to take relative to its current state, resulting in the greatest reward!</p>
            <p className="text-center flex-wrap">Source: <a target="_blank" href="https://www.analyticsvidhya.com/blog/2019/04/introduction-deep-q-learning-python/" rel="noopener noreferrer" className="text-yellow-400 underline hover:text-yellow-600">Analytic Vidhya</a></p>
          </div>
        </article>
        <article>
            <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <video autoPlay muted loop className="max-h-96">         
                <source src="/deepQ.mp4" type="video/mp4"/>       
            </video>
                {/* <Image alt="Video of our DeepQ agent running the Green Hill Zone level." src={deepQVideo} width={128} height={128}/> */}
            </div>
        </article>
        <article>
          <h2 className="text-center text-yellow-400 p-6 md:px-8 lg:px-16">Our Code:</h2>
          <div className="m-6 md:mx-8 lg:mx-16">
          <SyntaxHighlighter 
            showLineNumbers
            style={vscDarkPlus}
            language="python">
            {code} 
            </SyntaxHighlighter>
          </div>
        </article>
      </main>
    </>
  );
};

export default DeepQ;