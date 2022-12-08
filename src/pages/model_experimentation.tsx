import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";
import Image from "next/image";

import loop from "../../public/loopImg.jpg"


const Home: NextPage = () => {
  return (
    <>
      <Head>
        <title>Model Experimentation</title>
        <meta name="description" content="Exploring how to advance our model and developing NEAT and DeepQ agents" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Phase 2: Model Experimentation</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Introduction</h2>
            <Image className="mx-auto max-h-96 w-auto" alt="The first loop in Sonic that is the biggest barrier to advancement in level 1" src={loop} width={1244} height={818}></Image>
            <p className="text-center">As the team approached this problem with limited knowledge of reinforcement learning, we turned towards scholarly research to inform our decision on which models to include in experimentation. First, we implemented NEAT, NeuroEvolution of Augmenting Topologies, as our flagship model as it performs optimally on a CPU. As the team gained GPU processing capabilities, we transitioned to a more robust Deep Q Learning Agent. These agents were tuned in parallel to identify the best model for our situation. We ultimately moved forward with development of the Deep Q Learning Agent. Deep Q uses a neural network to calculate action-reward pairs for each input state in parallel, resulting in a policy that maps states to actions that obtain the greatest reward.</p>
            <p className="text-center">However, we found that the agent benefitted from seeing certain states more often than others to develop a policy that could overcome the trickiest of obstacles, such as the dreaded loop. As such, we developed the replay agent that supported training of the Deep Q Agent. This helper agent allows the user to control the starting point of the Deep Q Agent&apos;s training, allowing the agent to focus better on obstacles that would otherwise prove insurmountable during standard training. </p>
          </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/neat"}>NEAT -&gt;</Link>
          </button>
        </article>
      </main>
    </>
  );
};

export default Home;
