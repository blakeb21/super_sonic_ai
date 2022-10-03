import type { NextPage } from "next";
import Head from "next/head";
// import { trpc } from "../utils/trpc";
import Link from "next/link";
import Header from "../components/header";


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
            <h1 className="text-yellow-400 text-2xl m-2">Welcome to SuperSonicAI</h1>
            <p className="">SuperSonicAI is our semester long Capstone Project. Our goal was to expand on an AI competition to expand our knowledge and skills in the realm of Computer Science. </p>
          </div>
        </article>
        <article>
          <div className="container mx-auto flex flex-col p-6 md:px-8 lg:px-16">
            <h2 className="text-yellow-400 text-center">Phase 1: NEAT Implementation</h2>
            <p className="text-center">The first step in our project was to create a basic NEAT implementation. NEAT stands for NeuroEvolution of Augmenting Topologies, which is a form of evolutionary machine learning developed by researchers at MIT. The result is a fast but structured machine learning algorithm.</p>
            <Link href={'/neat'}>
              <button className="rounded bg-yellow-400 text-black mx-auto p-2">Learn More!</button>
            </Link>
          </div>
        </article>
      </main>
    </>
  );
};

export default Home;
