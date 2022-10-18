import type { NextPage } from "next";
import Head from "next/head";
import Header from "../components/header";
import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter';
import {vscDarkPlus} from 'react-syntax-highlighter/dist/cjs/styles/prism';


const ActorCritic: NextPage = () => {

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
            <h1 className="text-yellow-400 text-2xl m-2">Actor-Critic Implemenation</h1>
            <p className="text-center">The next step in our project was to create an Actor-Critic implementation. The Actor-Critic model is a form of reinforcement learning that seperates the value calculation from the policy decisions. The Actor, in this model, determines which policies to make and makes decisions based on those policies. The Critic then determines the estimated effects of the decisions made and provides feedback to the Actor through temporal difference errors.</p>
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

export default ActorCritic;