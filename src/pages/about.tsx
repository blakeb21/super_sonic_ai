import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Header from "../components/header";


const About: NextPage = () => {

  return (
    <>
      <Head>
        <title>About Us</title>
        <meta name="description" content="Learn about our development team and project." />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen">
        <Header />
        <article className="container mx-auto flex flex-col items-center p-4">
                <h1 className="text-purple-500 text-4xl m-4 text-center">Project Team:</h1>
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
            <article className="container mx-auto flex flex-col text-center items-center p-4">
                <h2 className="text-purple-500 text-2xl m-2">The Goal:</h2>
                <p className="p-2 md:mx-32">The goal of this project is to build on the original <a target="_blank" href="https://openai.com/blog/retro-contest/" rel="noopener noreferrer" className="text-yellow-400 underline hover:text-yellow-600">2018 OpenAI Retro Contest</a> that ran from April 5th, 2018 to June 5th, 2018. We will roughly follow the original guidelines and attempt to beat the winner from 2018.</p>
            </article>
            <article className="container mx-auto text-center flex flex-col p-4">
                <h3 className="text-purple-500 text-2xl m-2">Contest Description:</h3>
                <div className="p-2 md:mx-32">
                    <p>&quot;We&apos;re holding a transfer-learning contest using the Sonic The Hedgehog™ series of games for SEGA Genesis. In this contest, participants try to create the best agent for playing custom levels of the Sonic games — without having access to those levels during development. Here&apos;s how the contest works:</p>
                    <ol>
                        <li>1. Train or script your agent to play Sonic The Hedgehog™</li>
                        <li>2. Submit your agent to us as a Docker container</li>
                        <li>3. We evaluate your agent on a set of secret test levels</li>
                        <li>4. Your agent&apos;s score appears on the leaderboard&quot;</li>
                    </ol>    
                    <p className=""><a target="_blank" href="https://contest.openai.com/2018-1/" rel="noopener noreferrer" className="text-yellow-400 underline hover:text-yellow-600">- The OpenAI Team</a></p>
                </div>
        </article>
    </main>
    </>
  );
};

export default About;
