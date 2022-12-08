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
        <title>Model Development</title>
        <meta name="description" content="Exploring how to advance our model and developing NEAT and DeepQ agents" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article>
          <div className="container mx-auto flex flex-col text-center p-6 md:px-8 lg:px-16">
            <h1 className="text-yellow-400 text-2xl m-2">Phase 3: Model Experimentation</h1>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center m-1">Introduction</h2>
            <p className="text-center">This experimentation of RL models led us to a solution that can finish the first level of Sonic utilizing specifically tuned configurations of a Deep Q Learning Agent. The team then decided to tune our hyperparameters towards optimal generalization. We implemented a stochastic frame skipping to introduce randomness to the model and created several reward structures that better generalized to unseen environments. Further, we preprocessed our images using the OpenCV library to reduce our input dimensions to facilitate longer training across multiple levels. While these changes reduced performance on level 1, they align with real RL applications as the model now generalizes better to unseen environments.</p>
          </div>
        </article>
        <article className="container text-end">
          <button className="rounded bg-yellow-400 text-black p-2  mb-8 mt-8">
            <Link href={"/deepq_tuning"}>DeepQ Tuning -&gt;</Link>
          </button>
        </article>
      </main>
    </>
  );
};

export default Home;
